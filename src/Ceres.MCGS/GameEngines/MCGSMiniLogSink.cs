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
using System.Collections.Generic;
using System.IO;
using System.Threading;

#endregion

namespace Ceres.MCGS.GameEngines;

/// <summary>
/// Owns the physical file behind one or more MCGSMiniLog writers.
///
/// A tournament run with concurrency greater than one creates an independent engine instance per
/// concurrency slot, and every such instance for a given engine ID resolves to the same minilog
/// path. Rather than disambiguating by slot (which produced one partial log file per slot), all
/// those writers attach to a single shared sink here, exactly as the concurrent game threads all
/// append to a single shared PGN file.
///
/// Sinks are refcounted and held in a process-wide registry keyed by full path: the first writer to
/// ask for a path creates (and truncates) the file, later writers attach to the same instance, and
/// the file is closed — with its final HTML rendering produced — only when the last writer releases
/// it. Without the refcount the first slot to finish would close the file out from under the others.
///
/// This class owns only file I/O and the HTML regeneration worker; the per-writer staging of whole
/// games lives in MCGSMiniLog.
/// </summary>
internal sealed class MCGSMiniLogSink
{
  #region Registry

  /// <summary>
  /// All currently open sinks, keyed by full (normalized) path so that "./x" and "x" agree.
  /// </summary>
  static readonly Dictionary<string, MCGSMiniLogSink> sinks = new(StringComparer.Ordinal);

  /// <summary>
  /// Guards the registry. Lock ordering is always registryLock before an instance lockObj; the two
  /// are never taken in the opposite order (Release deliberately closes the file outside this lock,
  /// since closing joins the HTML worker and does a final synchronous render).
  /// </summary>
  static readonly object registryLock = new();


  /// <summary>
  /// Returns the sink for a given file, creating it (and truncating any existing file) if this is
  /// the first writer to ask for it, otherwise attaching to the sink already open for that path.
  /// </summary>
  /// <param name="fileName">full path of the minilog text file</param>
  /// <param name="expectedConcurrentWriters">number of writers expected to share this file (1 = exclusive)</param>
  /// <param name="isFirstAttach">true if this call created the sink rather than attaching to an existing one</param>
  internal static MCGSMiniLogSink Acquire(string fileName, int expectedConcurrentWriters, out bool isFirstAttach)
  {
    string key = Path.GetFullPath(fileName);

    lock (registryLock)
    {
      if (sinks.TryGetValue(key, out MCGSMiniLogSink existing))
      {
        if (expectedConcurrentWriters > 1)
        {
          // Expected case under tournament concurrency: share the one file.
          existing.refCount++;
          isFirstAttach = false;
          return existing;
        }

        // An exclusive writer must not join a shared file (its unstaged, per-move writes would
        // interleave with another writer's games). Only reachable when two engines independently
        // derive the same standalone auto-name; previously this threw IOException on the second
        // open. Fall back to the next free variant of the name.
        key = NextFreeKey(key);
        fileName = key;
      }

      MCGSMiniLogSink created = new MCGSMiniLogSink(fileName, expectedConcurrentWriters);
      created.refCount = 1;
      sinks.Add(key, created);
      isFirstAttach = true;
      return created;
    }
  }


  /// <summary>
  /// Finds the first unused "base.N.minilog.txt" variant of a path already present in the registry.
  /// Caller must hold registryLock.
  /// </summary>
  static string NextFreeKey(string key)
  {
    const string SUFFIX = ".minilog.txt";
    string stem = key.EndsWith(SUFFIX, StringComparison.OrdinalIgnoreCase)
                ? key.Substring(0, key.Length - SUFFIX.Length)
                : key;
    string tail = key.EndsWith(SUFFIX, StringComparison.OrdinalIgnoreCase) ? SUFFIX : ".minilog.txt";

    for (int i = 2; ; i++)
    {
      string candidate = stem + "." + i + tail;
      if (!sinks.ContainsKey(candidate))
      {
        return candidate;
      }
    }
  }


  /// <summary>
  /// Detaches one writer. The underlying file is flushed, closed and given its final HTML
  /// rendering only when the last writer detaches.
  /// </summary>
  internal void Release()
  {
    bool doClose;

    lock (registryLock)
    {
      refCount--;
      doClose = refCount <= 0;
      if (doClose)
      {
        sinks.Remove(Path.GetFullPath(fileName));
      }
    }

    if (doClose)
    {
      // Outside registryLock: this joins the HTML worker and renders synchronously.
      CloseFile();
    }
  }

  #endregion

  readonly string fileName;
  readonly StreamWriter writer;
  readonly object lockObj = new();
  readonly int expectedConcurrentWriters;

  int refCount;             // guarded by registryLock
  bool headerWritten;
  bool closed;

  // Background HTML regeneration. The (cheap, incremental) text log is written on the calling
  // (game-playing) thread, but the full HTML rebuild at each game end is offloaded to a dedicated
  // low-priority background thread so the main thread is never blocked and the rebuild never steals
  // cycles from search. Regenerations are coalesced: at most one runs at a time per log, and if
  // further game-ends arrive while one is in flight a single follow-up pass is queued, so the final
  // HTML always reflects the latest state and a slow render can never overwrite a newer one.
  readonly object htmlLock = new();
  Thread htmlThread;        // the currently running regeneration worker (null when idle)
  bool htmlRegenQueued;     // another regeneration was requested while the worker was running

  int gamesCompleted;       // number of completed game blocks written (guarded by lockObj)

  // For the first HTML_REGEN_EVERY_GAME_THRESHOLD games the HTML is regenerated after every game;
  // beyond that it is regenerated only every HTML_REGEN_THROTTLE_INTERVAL-th game, to bound the
  // cumulative rebuild cost over long tournaments. Close always produces the final complete HTML.
  // Under concurrency this counter spans all sharing writers, which is what is wanted: the rebuild
  // cost scales with the size of the (single) file, not with any one slot's share of it.
  const int HTML_REGEN_EVERY_GAME_THRESHOLD = 100;
  const int HTML_REGEN_THROTTLE_INTERVAL = 10;


  /// <summary>
  /// Name of the underlying text file being written.
  /// </summary>
  internal string FileName => fileName;


  /// <summary>
  /// Number of writers this file was opened to serve (1 = exclusive, unstaged writes).
  /// </summary>
  internal int ExpectedConcurrentWriters => expectedConcurrentWriters;


  MCGSMiniLogSink(string fileName, int expectedConcurrentWriters)
  {
    this.fileName = fileName ?? throw new ArgumentNullException(nameof(fileName));
    this.expectedConcurrentWriters = expectedConcurrentWriters;
    writer = new StreamWriter(fileName, append: false) { AutoFlush = false };
  }


  /// <summary>
  /// Writes the one-time header section. Returns true if this call actually wrote it; subsequent
  /// calls (from writers sharing the file) are ignored and return false.
  /// </summary>
  internal bool WriteHeaderOnce(string headerText)
  {
    lock (lockObj)
    {
      if (closed || headerWritten)
      {
        return false;
      }
      headerWritten = true;

      writer.Write(headerText);
      writer.Flush();
      return true;
    }
  }


  /// <summary>
  /// Atomically appends one already-formatted block of text and flushes, so the block cannot be
  /// interleaved with a block written concurrently by another writer sharing this file. This is the
  /// single write point for all minilog content other than the header.
  /// </summary>
  /// <param name="text">block to append (already newline-terminated)</param>
  /// <param name="completesGame">true if this block completes a game (drives HTML regeneration)</param>
  internal void AppendBlock(string text, bool completesGame)
  {
    lock (lockObj)
    {
      if (closed || string.IsNullOrEmpty(text))
      {
        return;
      }

      writer.Write(text);
      writer.Flush();

      if (!completesGame)
      {
        return;
      }

      gamesCompleted++;

      // Regenerate the standalone HTML rendering alongside the log so an up-to-date view is available.
      // Regenerate after every game until the throttle threshold, then only every Nth game thereafter
      // (the final, complete HTML is always produced by CloseFile regardless). The rebuild runs on a
      // low-priority background worker so the game-playing thread is never blocked by it.
      bool shouldRegenerate = gamesCompleted <= HTML_REGEN_EVERY_GAME_THRESHOLD
                           || (gamesCompleted % HTML_REGEN_THROTTLE_INTERVAL) == 0;
      if (shouldRegenerate)
      {
        RequestHtmlRegeneration();
      }
    }
  }


  /// <summary>
  /// Requests a background regeneration of the HTML rendering. If a regeneration is already running,
  /// a single follow-up pass is queued so the final output reflects the latest log state (rather than
  /// starting overlapping renders that could finish out of order). Never throws.
  /// </summary>
  void RequestHtmlRegeneration()
  {
    lock (htmlLock)
    {
      if (closed)
      {
        return; // CloseFile performs the final, authoritative regeneration.
      }

      if (htmlThread != null && htmlThread.IsAlive)
      {
        htmlRegenQueued = true; // coalesce into one follow-up pass after the current render
        return;
      }

      htmlRegenQueued = false;
      htmlThread = new Thread(RunHtmlRegenerationLoop)
      {
        IsBackground = true,             // never keep the process alive on this worker
        Priority = ThreadPriority.Lowest, // diagnostic-only; must not compete with search threads
        Name = "MCGSMiniLogHtml"
      };
      htmlThread.Start();
    }
  }


  /// <summary>
  /// Background worker: regenerates the HTML, then keeps going while follow-up passes were requested
  /// during the prior render. Exactly one instance of this loop runs at a time per sink, so renders
  /// never overlap and the last pass always observes the most recently written log state.
  /// </summary>
  void RunHtmlRegenerationLoop()
  {
    while (true)
    {
      try
      {
        MCGSMiniLogHtmlFormatter.WriteHtmlFile(fileName, MCGSMiniLog.HtmlFileNameFor(fileName));
      }
      catch (Exception)
      {
        // Never let HTML generation disrupt the tournament; just skip this pass.
      }

      lock (htmlLock)
      {
        if (!htmlRegenQueued)
        {
          htmlThread = null;
          return;
        }
        htmlRegenQueued = false; // consume the queued request and render again
      }
    }
  }


  /// <summary>
  /// Flushes and closes the underlying file, then produces the final HTML rendering.
  /// Called only when the last writer releases this sink.
  /// </summary>
  void CloseFile()
  {
    lock (lockObj)
    {
      if (closed)
      {
        return;
      }
      closed = true;
      writer.Flush();
      writer.Dispose();
    }

    // Wait for any in-flight background HTML regeneration to finish first (so it cannot overwrite
    // the file with a stale render after we exit), then do one final synchronous regeneration to
    // guarantee the HTML reflects the fully written log.
    Thread pending;
    lock (htmlLock)
    {
      pending = htmlThread;
    }
    try
    {
      pending?.Join();
    }
    catch (Exception)
    {
      // Ignore failures from the background regeneration.
    }
    try
    {
      MCGSMiniLogHtmlFormatter.WriteHtmlFile(fileName, MCGSMiniLog.HtmlFileNameFor(fileName));
    }
    catch (Exception)
    {
      // Ignore HTML generation failures.
    }
  }
}
