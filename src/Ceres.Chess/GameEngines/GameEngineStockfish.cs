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

using Ceres.Base.OperatingSystem;
using Ceres.Chess.Positions;
using Ceres.Chess.UserSettings;

#endregion

namespace Ceres.Chess.GameEngines
{
  /// <summary>
  /// Convenience construction of Stockfish engines, and Stockfish's own win/draw/loss model.
  ///
  /// Deliberately a factory rather than a subclass of GameEngineUCI: nothing is missing from
  /// GameEngineUCI (its constructor already emits setoption for Threads/Hash/SyzygyPath), and
  /// every other part of Ceres creates engines through a GameEngineUCISpec (GameEngineDefUCI,
  /// GameEngineDefFactory, tournaments). A subclass would simply be bypassed by that machinery,
  /// whereas a configured spec composes with all of it.
  /// </summary>
  public static class GameEngineStockfish
  {
    /// <summary>
    /// Default number of threads. One is right for the common case of many engines run in
    /// parallel over independent positions; raise it for a single interactive engine.
    /// </summary>
    public const int DEFAULT_NUM_THREADS = 1;

    /// <summary>
    /// Default transposition table size in megabytes.
    /// </summary>
    public const int DEFAULT_HASH_MB = 256;

    /// <summary>
    /// Position used by VerifyReportsWDL. Any position with a decisive, non-trivial evaluation
    /// works; this one is far enough from equal that a functioning engine reports a lopsided WDL.
    /// </summary>
    const string WDL_CHECK_FEN = "r1bqkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 10";


    /// <summary>
    /// Returns a GameEngineUCISpec configured for Stockfish.
    /// </summary>
    /// <param name="name">identifying name of engine</param>
    /// <param name="exePath">path to executable, or null to resolve automatically</param>
    /// <param name="numThreads">threads (default 1)</param>
    /// <param name="hashSizeMB">hash size in megabytes (default 256)</param>
    /// <param name="syzygyPath">tablebase path, or null to take the Ceres.json default</param>
    /// <param name="showWDL">if UCI_ShowWDL should be enabled (default true)</param>
    /// <param name="resetGameBetweenMoves">if ucinewgame should precede each position</param>
    /// <param name="extraUCIOptions">additional setoption command lines</param>
    /// <param name="processorGroupID"></param>
    public static GameEngineUCISpec Spec(string name = "SF",
                                         string exePath = null,
                                         int? numThreads = DEFAULT_NUM_THREADS,
                                         int? hashSizeMB = DEFAULT_HASH_MB,
                                         string syzygyPath = null,
                                         bool showWDL = true,
                                         bool resetGameBetweenMoves = false,
                                         List<string> extraUCIOptions = null,
                                         int processorGroupID = 0)
    {
      List<string> options = extraUCIOptions == null ? new List<string>() : new List<string>(extraUCIOptions);
      if (showWDL)
      {
        // Makes Stockfish report its own win/draw/loss alongside the centipawn score,
        // which GameEngineUCI surfaces as GameEngineSearchResult.WDL.
        options.Add("setoption name UCI_ShowWDL value true");
      }

      // Settings.TablebaseDirectory (not SyzygyPath) because it already reconciles the
      // SyzygyPath/DirTablebases duality and throws if both are specified.
      syzygyPath ??= CeresUserSettingsManager.Settings.TablebaseDirectory;

      return new GameEngineUCISpec(name, ResolveEXEPath(exePath),
                                   numThreads, hashSizeMB, syzygyPath,
                                   resetGameBetweenMoves: resetGameBetweenMoves,
                                   uciSetOptionCommands: options.Count == 0 ? null : options,
                                   processorGroupID: processorGroupID);
    }


    /// <summary>
    /// Returns a ready-to-use Stockfish engine (see Spec for argument documentation).
    /// </summary>
    public static GameEngineUCI Create(string name = "SF",
                                       string exePath = null,
                                       int? numThreads = DEFAULT_NUM_THREADS,
                                       int? hashSizeMB = DEFAULT_HASH_MB,
                                       string syzygyPath = null,
                                       bool showWDL = true,
                                       bool resetGameBetweenMoves = false,
                                       List<string> extraUCIOptions = null,
                                       int processorGroupID = 0)
      => Spec(name, exePath, numThreads, hashSizeMB, syzygyPath, showWDL,
              resetGameBetweenMoves, extraUCIOptions, processorGroupID).CreateEngine();


    /// <summary>
    /// Locates the Stockfish executable, searching in order:
    ///   1. the explicit path argument
    ///   2. the StockfishPath entry in Ceres.json
    ///   3. the PATH environment variable
    ///   4. conventional install locations for the platform
    /// Throws listing everything tried if not found (this is the first thing a new user hits,
    /// so the failure has to say what to fix rather than just which file was missing).
    /// </summary>
    public static string ResolveEXEPath(string explicitPath = null)
    {
      List<string> tried = new();

      if (explicitPath != null)
      {
        // An explicit path that does not exist is a caller error, not a reason to fall back
        // to some other engine binary the caller did not ask for.
        if (!File.Exists(explicitPath))
        {
          throw new FileNotFoundException($"Stockfish executable not found at specified path {explicitPath}");
        }
        return explicitPath;
      }

      string fromSettings = CeresUserSettingsManager.Settings.StockfishPath;
      if (!string.IsNullOrEmpty(fromSettings))
      {
        if (File.Exists(fromSettings))
        {
          return fromSettings;
        }
        tried.Add($"{fromSettings} (StockfishPath in Ceres.json)");
      }

      string exeName = SoftwareManager.IsLinux ? "stockfish" : "stockfish.exe";

      string pathVar = Environment.GetEnvironmentVariable("PATH");
      if (pathVar != null)
      {
        foreach (string dir in pathVar.Split(Path.PathSeparator))
        {
          if (dir.Length == 0)
          {
            continue;
          }

          string candidate = Path.Combine(dir, exeName);
          if (File.Exists(candidate))
          {
            return candidate;
          }
        }
      }
      tried.Add($"{exeName} on PATH");

      foreach (string candidate in DefaultInstallPaths(exeName))
      {
        if (File.Exists(candidate))
        {
          return candidate;
        }
        tried.Add(candidate);
      }

      throw new FileNotFoundException("Unable to locate a Stockfish executable. Set StockfishPath in "
                                    + "Ceres.json, or pass an explicit path. Tried: " + string.Join(", ", tried));
    }


    static IEnumerable<string> DefaultInstallPaths(string exeName)
    {
      if (SoftwareManager.IsLinux)
      {
        yield return "/usr/games/" + exeName;
        yield return "/usr/bin/" + exeName;
        yield return "/usr/local/bin/" + exeName;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        if (!string.IsNullOrEmpty(home))
        {
          yield return Path.Combine(home, "bin", exeName);
        }
      }
      else
      {
        string dirExternal = CeresUserSettingsManager.Settings.DirExternalEngines;
        if (!string.IsNullOrEmpty(dirExternal))
        {
          yield return Path.Combine(dirExternal, exeName);
        }
      }
    }


    /// <summary>
    /// Verifies the engine actually reports WDL, throwing if not.
    ///
    /// This is not merely defensive: UCIGameRunner does not wait for uciok before sending the
    /// setoption commands, so a slow-starting engine can silently miss UCI_ShowWDL and then
    /// yield NaN evaluations for an entire run.
    /// </summary>
    public static void VerifyReportsWDL(GameEngine engine, int nodes = 10_000)
    {
      GameEngineSearchResult check = engine.Search(PositionWithHistory.FromFENAndMovesUCI(WDL_CHECK_FEN),
                                                   SearchLimit.NodesPerMove(nodes));
      if (!check.HasWDL)
      {
        throw new Exception($"Engine {engine.ID} did not report WDL. An engine supporting the "
                          + "UCI_ShowWDL option is required (and it must be enabled, see GameEngineStockfish.Spec).");
      }
    }
  }


  /// <summary>
  /// Stockfish's own win/draw/loss model, for recovering W/D/L from a stored centipawn score
  /// when no engine is available to ask.
  ///
  /// Prefer the engine's own reported WDL (UCI_ShowWDL, enabled by default in
  /// GameEngineStockfish.Spec) whenever an engine is running -- that is exact, and this model
  /// is only an approximation of it. This exists for offline reconstruction from archived data.
  ///
  /// Modern Stockfish parameterises WDL by MATERIAL rather than by ply. Grouping a live sample
  /// by centipawns alone leaves a draw-probability spread of 0.0436; grouping by
  /// (centipawns, material) collapses it to 0.00067. The model then reduces to a single scalar
  /// alpha = a/b per material value:
  ///
  ///     W = sigmoid(alpha * (cp/100 - 1))
  ///     L = sigmoid(-alpha * (cp/100 + 1))
  ///     D = 1 - W - L
  /// </summary>
  public static class StockfishWDL
  {
    /// <summary>
    /// Stockfish version the alpha table was fitted against. A later Stockfish may change its
    /// WDL model, in which case this table becomes silently wrong -- refit with
    /// CeresTrain's Remix_V2/fit_alpha.py and update ALPHA_BY_MATERIAL.
    /// </summary>
    public const string FITTED_FOR = "Stockfish 18";

    /// <summary>
    /// Lowest material value covered by the table (values below are clamped to it).
    /// </summary>
    public const int MIN_MATERIAL = 17;

    /// <summary>
    /// Highest material value covered by the table (values above are clamped to it).
    /// </summary>
    public const int MAX_MATERIAL = 78;

    /// <summary>
    /// Fitted alpha indexed by (material - MIN_MATERIAL), from 40,960 live Stockfish samples.
    /// Round trip (encode to Half, invert, apply model) reproduces Stockfish's own W/D/L to a
    /// mean absolute Q error of 0.00075, which is the per-mille rounding floor of its output.
    /// </summary>
    static readonly float[] ALPHA_BY_MATERIAL =
    [
      6.623732f, 6.595400f, 6.560144f, 6.533023f, 6.516671f, 6.495189f,
      6.486630f, 6.476119f, 6.460987f, 6.460529f, 6.454684f, 6.453430f,
      6.463011f, 6.459410f, 6.461300f, 6.472413f, 6.470280f, 6.467115f,
      6.479648f, 6.483236f, 6.491273f, 6.495342f, 6.504030f, 6.500280f,
      6.503202f, 6.485438f, 6.489072f, 6.474150f, 6.462911f, 6.441795f,
      6.419823f, 6.402866f, 6.370370f, 6.341514f, 6.295400f, 6.248513f,
      6.198953f, 6.150762f, 6.091269f, 6.032395f, 5.964297f, 5.892085f,
      5.817719f, 5.733140f, 5.648596f, 5.561305f, 5.468369f, 5.371632f,
      5.272559f, 5.168802f, 5.066550f, 4.960534f, 4.849057f, 4.739478f,
      4.629306f, 4.517293f, 4.404543f, 4.289438f, 4.174903f, 4.061840f,
      3.948649f, 3.835655f,
    ];


    /// <summary>
    /// Returns Stockfish's modelled (W, D, L) for a centipawn score at a given material total.
    /// </summary>
    /// <param name="centipawns">score from the side to move's perspective</param>
    /// <param name="material">sum of piece values (P1 N3 B3 R5 Q9), clamped into range</param>
    public static (float W, float D, float L) FromCentipawns(float centipawns, int material)
    {
      int index = Math.Clamp(material, MIN_MATERIAL, MAX_MATERIAL) - MIN_MATERIAL;
      float alpha = ALPHA_BY_MATERIAL[index];

      float x = centipawns / 100.0f;
      float w = Sigmoid(alpha * (x - 1));
      float l = Sigmoid(-alpha * (x + 1));

      // W and L are modelled independently, so their sum can marginally exceed 1 at extremes.
      float d = 1.0f - w - l;
      if (d < 0)
      {
        float sum = w + l;
        return (w / sum, 0, l / sum);
      }

      return (w, d, l);
    }


    /// <summary>
    /// Returns Stockfish's modelled (W, D, L) for a centipawn score at a given position.
    /// </summary>
    public static (float W, float D, float L) FromCentipawns(float centipawns, in Position position)
      => FromCentipawns(centipawns, MaterialOf(in position));


    /// <summary>
    /// Returns the material total (P1 N3 B3 R5 Q9, kings excluded) used to index the model.
    /// </summary>
    public static int MaterialOf(in Position position)
    {
      int material = 0;
      foreach ((Piece piece, Square _) in position)
      {
        material += piece.Type switch
        {
          PieceType.Pawn => 1,
          PieceType.Knight => 3,
          PieceType.Bishop => 3,
          PieceType.Rook => 5,
          PieceType.Queen => 9,
          _ => 0
        };
      }
      return material;
    }


    /// <summary>
    /// Converts a logistic value written BEFORE commit 2df6a86e (2026-06-22) back to centipawns.
    ///
    /// That commit fixed CentipawnToLogistic, which had applied a spurious atanh:
    ///   old:  logistic = atanh(atan(cp/90) / 1.5637541897)
    ///   new:  logistic =       atan(cp/90) / 1.5637541897     (the true inverse of
    ///                                                          LogisticToCentipawn)
    /// so data written before that date must be inverted with the old form, which this does.
    /// Using EncodedEvalLogistic.LogisticToCentipawn on such data yields wrong centipawns
    /// silently -- hence a separate, explicitly named entry point rather than a flag.
    ///
    /// Note the old form saturates: above roughly 225 centipawns the stored value is pinned
    /// near 1.0 and the original magnitude is unrecoverable. That costs little in practice,
    /// because Stockfish is already at least 0.99 decisive there.
    /// </summary>
    public static float CentipawnFromLegacyLogistic(float logistic)
    {
      const float CENTIPAWN_MULT = 90.0f;
      const float CENTIPAWN_TAN_MULT = 1.5637541897f;

      float bounded = Math.Clamp(logistic, -0.999999f, 0.999999f);
      return CENTIPAWN_MULT * MathF.Tan(CENTIPAWN_TAN_MULT * MathF.Tanh(bounded));
    }


    static float Sigmoid(float x) => 1.0f / (1.0f + MathF.Exp(-x));
  }
}
