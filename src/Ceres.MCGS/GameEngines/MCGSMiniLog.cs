#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directives

using System;
using System.Globalization;
using System.IO;
using System.Text;

using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.NNEvaluators.Defs;

using Ceres.MCGS.UCI;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.GameEngines;

/// <summary>
/// Writer for a per-tournament diagnostic "minilog" text file emitted by a
/// GameEngineCeresMCGSInProcess. The file is intended primarily for consumption by
/// post-processing tools, with brief parser-friendly labels.
///
/// File structure (single file for the whole tournament):
///   - a one-time header (timestamp, host/system info, engine configuration dumps),
///   - per game: a separator line, then one line per move,
///   - per game: a result footer block supplied by the caller.
///
/// This is a lightweight per-engine handle; the file itself is owned by a shared, refcounted
/// MCGSMiniLogSink. When a tournament runs several concurrent game threads, each thread has its own
/// engine instance and hence its own handle, but all handles for a given engine ID attach to one
/// sink and therefore produce one aggregated file (mirroring the single aggregated PGN).
///
/// Because concurrent games would otherwise interleave their move lines — corrupting the per-game
/// structure that the HTML renderer parses positionally — a shared handle stages the whole current
/// game in memory and appends it to the file as one atomic block when the game ends. This is
/// precisely how a game is written to the shared PGN (buffered in a PGNWriter, then appended under a
/// lock). An exclusive handle (concurrency of one, and the standalone/UCI path) skips staging and
/// writes through immediately with a flush per line, so the most recent move still survives an
/// abnormal termination.
/// </summary>
public sealed class MCGSMiniLog : IDisposable
{
  /// <summary>
  /// Order of the scalar tokens emitted on each move line (documented in the header legend).
  /// </summary>
  public const string BODY_LEGEND =
    "LEGEND (per move line): FEN, RootN, StoreN, NNEvals, TimeRem, OppTimeRem, LimInit, Elapsed, "
    + "BudgetFrac%, NPS, EPS, BackendBusy, Depth, SelDepth | candidate moves as (SAN, visit%, Q) "
    + "sorted by visits descending, '*' prefixes the played move, Q from side-to-move perspective. "
    + "Sel (present only when the played move was not the most-visited move) names the selection "
    + "mechanism: best-Q / minimax / irreversible / drp-avoid. "
    + "TimeRem / OppTimeRem are the engine's own and the opponent's remaining game clock (seconds), "
    + "populated only for SecondsForAllMoves play within a Ceres tournament (else n/a). "
    + "LimInit is the per-move allocated budget (seconds for time limits, nodes for node limits); "
    + "BudgetFrac is the percent of that budget used this move (elapsed/LimInit for time limits, "
    + "nodes-this-search/LimInit for node limits). "
    + "Note: RootN/visit% are cumulative across tree reuse, so per-move visit% may sum to under 100.";

  /// <summary>
  /// Marker appended to a staged game block that was flushed without a result footer
  /// (for example because the tournament was aborted part way through the game).
  /// </summary>
  const string INCOMPLETE_MARKER = "=== GAME INCOMPLETE ===";

  /// <summary>
  /// Line terminator, matching what StreamWriter.WriteLine and StringBuilder.AppendLine emit
  /// (qualified because the unqualified name binds to the Ceres.MCGS.Environment namespace).
  /// </summary>
  static readonly string NL = System.Environment.NewLine;

  readonly MCGSMiniLogSink sink;

  /// <summary>
  /// If whole games are staged in memory and appended atomically (required when this file is shared
  /// with the engine instances of other concurrent game threads). Decided once, up front, from the
  /// tournament's declared concurrency — never from a live writer count, which would race as the
  /// concurrency slots attach at staggered times.
  /// </summary>
  readonly bool stageWholeGames;

  readonly object bufLock = new();
  StringBuilder pending;    // staged text for the current game (staging mode only)
  bool closed;              // guarded by bufLock


  /// <summary>
  /// Name of the underlying text file being written.
  /// </summary>
  public string FileName => sink.FileName;


  /// <summary>
  /// Name of the companion HTML file rendered alongside the text minilog.
  /// </summary>
  public string HtmlFileName => HtmlFileNameFor(sink.FileName);


  /// <summary>
  /// True if this handle created the underlying file (rather than attaching to a file already
  /// opened by another concurrent game thread). Useful to emit a console notice exactly once.
  /// </summary>
  public bool IsPrimaryWriter { get; }


  /// <summary>
  /// Derives the companion HTML file name for a given minilog text file name. The text minilog
  /// ends in ".minilog.txt"; the HTML rendering replaces the trailing ".txt" with ".html" so the
  /// companion is named "base.minilog.html" (rather than "base.minilog.txt.html").
  /// </summary>
  public static string HtmlFileNameFor(string logFileName)
  {
    if (logFileName != null && logFileName.EndsWith(".txt", StringComparison.OrdinalIgnoreCase))
    {
      return logFileName.Substring(0, logFileName.Length - 4) + ".html";
    }
    return logFileName + ".html";
  }


  /// <summary>
  /// Opens (or attaches to) the minilog for a given file.
  /// </summary>
  /// <param name="fileName">full path of the file to write</param>
  /// <param name="expectedConcurrentWriters">
  /// number of engine instances expected to write to this file (the tournament's concurrency).
  /// A value greater than one attaches to a shared file and enables whole-game staging.
  /// </param>
  public static MCGSMiniLog Acquire(string fileName, int expectedConcurrentWriters = 1)
  {
    ArgumentNullException.ThrowIfNull(fileName);

    MCGSMiniLogSink sink = MCGSMiniLogSink.Acquire(fileName, expectedConcurrentWriters, out bool isFirstAttach);
    return new MCGSMiniLog(sink, expectedConcurrentWriters > 1, isFirstAttach);
  }


  MCGSMiniLog(MCGSMiniLogSink sink, bool stageWholeGames, bool isPrimaryWriter)
  {
    this.sink = sink;
    this.stageWholeGames = stageWholeGames;
    IsPrimaryWriter = isPrimaryWriter;
  }


  /// <summary>
  /// Appends already-formatted text: staged into the current game block when sharing the file with
  /// other concurrent writers, otherwise written straight through (with an immediate flush).
  /// </summary>
  /// <param name="text">text to emit (already newline-terminated)</param>
  /// <param name="completesGame">true if this completes the current game block</param>
  void Emit(string text, bool completesGame)
  {
    lock (bufLock)
    {
      EmitLocked(text, completesGame);
    }
  }


  /// <summary>
  /// Body of Emit. Caller must hold bufLock.
  /// </summary>
  void EmitLocked(string text, bool completesGame)
  {
    if (closed)
    {
      return;
    }

    if (!stageWholeGames)
    {
      sink.AppendBlock(text, completesGame);
      return;
    }

    pending ??= new StringBuilder(64 * 1024);
    pending.Append(text);

    if (completesGame)
    {
      sink.AppendBlock(pending.ToString(), true);
      pending.Clear();
    }
  }


  /// <summary>
  /// Flushes any staged (incomplete) game block, tagging it so consumers can tell it was cut short.
  /// Caller must hold bufLock.
  /// </summary>
  void FlushIncompleteLocked()
  {
    if (pending == null || pending.Length == 0)
    {
      return;
    }

    pending.AppendLine(INCOMPLETE_MARKER);
    sink.AppendBlock(pending.ToString(), false);
    pending.Clear();
  }


  /// <summary>
  /// Writes the one-time header section (idempotent; ignored once written, including when written
  /// by another engine instance sharing this file). Includes timestamp, host/system information,
  /// engine identity and evaluator, the assigned search limit, and a full dump of the search and
  /// select parameters.
  /// </summary>
  public void WriteHeader(string engineID, NNEvaluatorDef evaluatorDef, SearchLimit assignedSearchLimit,
                          ParamsSearch searchParams, ParamsSelect selectParams)
  {
    lock (bufLock)
    {
      if (closed)
      {
        return;
      }
    }

    StringWriter writer = new StringWriter();

    writer.WriteLine("=== CERES MCGS MINILOG ===");
    writer.WriteLine("Timestamp: " + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture));

    // Host / OS / runtime / process information (reuse existing diagnostics helper).
    DiagnosticsBlock.WriteSystemInfoHeader(writer);

    writer.WriteLine("Engine    : " + engineID);
    writer.WriteLine("Evaluator : " + (evaluatorDef == null ? "(none)" : evaluatorDef.ToString()));
    writer.WriteLine("AssignedSearchLimit: " + (assignedSearchLimit == null ? "(none)" : assignedSearchLimit.ToString()));

    if (searchParams != null)
    {
      writer.WriteLine();
      writer.WriteLine("--- ParamsSearch (all properties) ---");
      writer.Write(ObjUtils.FieldValuesDumpString<ParamsSearch>(searchParams, new ParamsSearch(), false));

      writer.WriteLine("--- ParamsSearchExecution (all properties) ---");
      writer.Write(ObjUtils.FieldValuesDumpString<ParamsSearchExecution>(searchParams.Execution, new ParamsSearchExecution(), false));

      writer.WriteLine("--- ParamsRootMinimaxBlend (all properties) ---");
      writer.Write(ObjUtils.FieldValuesDumpString<ParamsRootMinimaxBlend>(searchParams.RootMinimaxBlend, new ParamsRootMinimaxBlend(), false));
    }

    if (selectParams != null)
    {
      writer.WriteLine("--- ParamsSelect (all properties) ---");
      writer.Write(ObjUtils.FieldValuesDumpString<ParamsSelect>(selectParams, new ParamsSelect(), false));
    }

    writer.WriteLine();

    // Note the aggregation, so a reader knows why games from several threads appear interleaved
    // (and that the configuration above is that of whichever thread initialized first).
    if (sink.ExpectedConcurrentWriters > 1)
    {
      writer.WriteLine($"ConcurrentWriters: {sink.ExpectedConcurrentWriters} (games played by "
                     + $"{sink.ExpectedConcurrentWriters} concurrent tournament threads are aggregated "
                     + "into this file, in order of completion)");
    }

    writer.WriteLine(BODY_LEGEND);
    writer.WriteLine();

    sink.WriteHeaderOnce(writer.ToString());
  }


  /// <summary>
  /// Writes a separator marking the start of a new game.
  /// </summary>
  public void WriteNewGameSeparator(string gameID)
  {
    lock (bufLock)
    {
      if (closed)
      {
        return;
      }

      // Defensive: bound the staging buffer to a single game even if a result footer is ever missed.
      FlushIncompleteLocked();

      EmitLocked(NL + "=== NEW GAME: " + (gameID ?? "(unnamed)") + " ===" + NL, false);
    }
  }


  /// <summary>
  /// Writes a single (already formatted) per-move body line.
  /// </summary>
  public void WriteMoveLine(string line)
  {
    Emit(line + NL, false);
  }


  /// <summary>
  /// Appends a (preformatted) per-game result footer block. This completes the game: when staging,
  /// the whole accumulated game block is appended to the file atomically here.
  /// </summary>
  public void AppendGameResultFooter(string footerText)
  {
    Emit(footerText + NL, true);
  }


  /// <summary>
  /// Appends a (preformatted) limits-manager diagnostics block, delimited by markers so post-processors
  /// can locate it. Emitted inline immediately before the move whose budget it describes
  /// (the allocation is computed before the move is searched), so it appears next to that move header.
  /// </summary>
  public void AppendLimitsSection(string limitsText)
  {
    if (string.IsNullOrEmpty(limitsText))
    {
      return;
    }

    Emit("=== LIMITS ===" + NL
       + limitsText + NL
       + "=== END LIMITS ===" + NL, false);
  }


  /// <summary>
  /// Appends a (preformatted) blunder-diagnostics block, delimited by markers so post-processors can
  /// locate it. Emitted inline after the move that triggered the blunder confirmation.
  /// </summary>
  public void AppendBlunderSection(string blunderText)
  {
    Emit("=== BLUNDER ===" + NL
       + blunderText + NL
       + "=== END BLUNDER ===" + NL, false);
  }


  /// <summary>
  /// Flushes any staged partial game and detaches from the underlying file (idempotent).
  /// The file is closed, and its final HTML rendering produced, once the last writer detaches.
  /// </summary>
  public void Close()
  {
    lock (bufLock)
    {
      if (closed)
      {
        return;   // must be exactly once: a second release would close the file under sibling writers
      }
      closed = true;

      FlushIncompleteLocked();
    }

    sink.Release();
  }


  void IDisposable.Dispose() => Close();
}
