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
using Ceres.Base.Misc;
using Ceres.Base.Threading;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.PositionEvalCaching;
using Ceres.Chess.Positions;
using Ceres.Chess.UserSettings;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Graphs.GraphStores;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Managers.Limits;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Phases.Evaluation;

using static Ceres.MCGS.Search.Coordination.MCGSManager;
using static Ceres.MCGS.Search.Phases.MCGSSelect;

#endregion

namespace Ceres.MCGS.Search.Coordination;

/// <summary>
/// Entry point for launching a search and capturing the results.

/// Note that in most situations instead the class GameEngineCeresInProcess 
/// is preferrable as the entry point for launching searches.
/// </summary>
public partial class MCGSSearch
{
  /// <summary>
  /// Optional delegate that registers to receive informational messages that should be logged.
  /// </summary>
  /// <param name="infoMessage"></param>
  public delegate void MCGSInfoLogger(string infoMessage);

  /// <summary>
  /// The underlying serach manager.
  /// </summary>
  public MCGSManager Manager { get; internal set; }

  /// <summary>
  /// Selected best move from last search.
  /// </summary>
  public MGMove BestMove { get; private set; }

  /// <summary>
  /// Node within the graph from which the search starts.
  /// </summary>
  public GNode SearchRootNode => Manager.Engine.SearchRootNode;

  /// <summary>
  /// N of the SearchRootNode when at beginning of search.
  /// </summary>
  public int StartSearchN { get; private set; }

  /// <summary>
  /// Total number of searches conducted.
  /// </summary>
  public static int SearchCount { get; internal set; }


  #region Graph reuse related

  /// <summary>
  /// The number of times a search from this tree
  /// has been satisfied out of tree reuse (no actual search).
  /// </summary>
  public int CountSearchContinuations { get; private set; }


  /// <summary>
  /// Optional delegate that registers to receive informational messages that should be logged.
  /// </summary>
  /// <param name="search"></param>
  /// <param name="infoMessage"></param>
  public readonly MCGSInfoLogger InfoLogger;

  #endregion


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="infoLogger"></param>
  public MCGSSearch(MCGSInfoLogger infoLogger = null)
  {
    InfoLogger = infoLogger;
  }

  /// <summary>
  /// Runs a new search.
  /// </summary>
  /// <param name="nnEvaluators"></param>
  /// <param name="graphToPossiblyReuse"></param>
  /// <param name="paramsSelect"></param>
  /// <param name="paramsSearch"></param>
  /// <param name="limitManager"></param>
  /// <param name="priorMoves"></param>
  /// <param name="searchLimit"></param>
  /// <param name="verbose"></param>
  /// <param name="startTime"></param>
  /// <param name="gameMoveHistory"></param>
  /// <param name="progressCallback"></param>
  /// <param name="isFirstMoveOfGame"></param>
  /// <param name="fixedSearchLimit"></param>
  public void Search(NNEvaluatorSet nnEvaluators,
                     Graph graphToPossiblyReuse,
                     WorkerPool<ExtendPathsWorkerInfo>[] selectWorkerPools,
                     ParamsSelect paramsSelect,
                     ParamsSearch paramsSearch,
                     IManagerGameLimit limitManager,
                     PositionWithHistory priorMoves,
                     SearchLimit searchLimit,
                     bool verbose,
                     DateTime startTime,
                     List<GameMoveStat> gameMoveHistory,
                     MCGSProgressCallback progressCallback = null,
                     PositionEvalCache positionEvalCache = null,
                     bool isFirstMoveOfGame = false,
                     bool moveImmediateIfOnlyOneMove = false,
                     MGMove forcedMove = default,
                     SearchLimit fixedSearchLimit = null)
  {
    if (searchLimit == null)
    {
      throw new ArgumentNullException(nameof(searchLimit));
    }

    if (searchLimit.SearchCanBeExpanded && !MCGSParamsFixed.STORAGE_USE_INCREMENTAL_ALLOC)
    {
      throw new Exception("STORAGE_USE_INCREMENTAL_ALLOC must be true when SearchCanBeExpanded.");
    }

    if (!MCGSParamsFixed.STORAGE_USE_INCREMENTAL_ALLOC && !searchLimit.IsNodesLimit)
    {
      throw new Exception("SearchLimit must be NodesPerMove or NodesPerGame when STORAGE_USE_INCREMENTAL_ALLOC is false");
    }

    paramsSearch.Validate();
    paramsSelect.Validate();
    paramsSelect.ValidateAgainst(paramsSearch);

    // TODO: clean this up?
    string tablebasePaths = paramsSearch.EnableTablebases ? paramsSearch.TablebasePaths : null;
    bool forceNoTablebaseTerminals = EvaluatorSyzygy.PosIsTablebaseWinWithNoDTZAvailable(tablebasePaths, priorMoves);

#if NOT
In MCGS version we build Sygyzy evaluator but tell it to instead return
neural network eval because WDL and not DTZ available
good example position: 8/5k2/8/6P1/8/7P/pBp1K3/8 b - - 1 51


    else
    {
      node.InfoRef.EvalResultAuxilliary = (FP16)result.V;
      return default;
    }

This isn't currently easy in the MCGS engine.
As a workaround, EvaluatorSygyzy will just return as if no hit.

#endif

#if NOT
// Have to disable this to avoid overflows
// The problem is probably that the sizing here doesn't account for the fact
// that the graph can grow thru reuse very large.
// (Superseded by the live store sizing below; the move overhead reserve that used to be
//  computed here is now applied to the per-move limit, see AdjustedPerMoveSearchLimit.)

    int maxNodes;
    if (!searchLimit.SearchCanBeExpanded && searchLimit.IsNodesLimit)
    {
      maxNodes = (int)(searchLimit.Value + searchLimit.ValueIncrement + 5000);
    }
    else
    {
      // In this mode, we are just reserving virtual address space
      // from a very large pool (e.g. 256TB for Windows).
      // Therefore it is safe to reserve a very large block.
      if (searchLimit.MaxTreeNodes != null)
      {
        // Reseve somewhat more storage than the maximum requested tree nodes
        // if the search can be expenaded because during tree rewrite
        // a preparatory step (MaterializeNodesWithNonRetainedTranspositionRoots)
        // will initially make the store larger (before it is subsequently compacted).
        double NODES_BUFFER_MULTIPLIER = searchLimit.SearchCanBeExpanded ? 1.2 : 1.0;
        long maxNodesLong = (long)(NODES_BUFFER_MULTIPLIER * searchLimit.MaxTreeNodes.Value) + 100_000;
        maxNodes = (int)Math.Min(maxNodesLong, int.MaxValue - 100_000);
      }
      else
      {
        maxNodes = GraphStore.MAX_NODES;
      }
    }
#endif

    // Preserve any cross-graph evaluation-reuse provider (e.g. opponent graph reuse) so it can be
    // re-established on a freshly created graph if the prior graph is abandoned below.
    Func<Graph> priorReuseGraphProvider = graphToPossiblyReuse?.ReuseGraphProvider;

    // Try to reuse the prior graph
    Graph graphToReuse = GraphReuseManager.TryReuseGraph(paramsSearch, priorMoves, graphToPossiblyReuse,
                                                        out GraphRootToSearchRootNodeInfo searchRootNodeInfo,
                                                        out List<GraphRootToSearchRootNodeInfo> searchRootPathFromGraphRoot);

    SearchLimit searchLimitToUse;
    ManagerGameLimitInputs gameLimitsInputs = null;
    ManagerGameLimitOutputs gameLimitsOutputs = null;
    if (!searchLimit.IsPerGameLimit)
    {
      searchLimitToUse = searchLimit;
    }
    else
    {
      SearchLimitType targetType = searchLimit.IsTimeLimit ? SearchLimitType.SecondsPerMove
                                                           : SearchLimitType.NodesPerMove;

      int searchRootNodeN = searchRootNodeInfo == default ? 0 : searchRootNodeInfo.ChildNode.N;
      double searchRootNodeQ = searchRootNodeInfo == default ? 0 : searchRootNodeInfo.ChildNode.Q;
      gameLimitsInputs = new(priorMoves.FinalPosition,
                           paramsSearch, gameMoveHistory,
                           targetType, searchRootNodeN, (float)searchRootNodeQ,
                           ClockAfterMoveOverheadReserve(searchLimit, paramsSearch), searchLimit.ValueIncrement,
                           searchLimit.MaxTreeNodes, searchLimit.MaxTreeVisits,
                           float.NaN, float.NaN,
                           maxMovesToGo: searchLimit.MaxMovesToGo,
                           isFirstMoveOfGame: isFirstMoveOfGame, paramsSearch.EnableQuickMoves);

      gameLimitsOutputs = limitManager.ComputeMoveAllocation(gameLimitsInputs);
      searchLimitToUse = gameLimitsOutputs.LimitTarget;
    }

    // Hold back the configured move overhead (UCI MoveOverheadMs) from the time actually searched.
    // For a true per-move limit (e.g. "go movetime") the overhead is reserved from this move's
    // budget, since the move is expected back within that budget. For a per-game clock the reserve
    // was instead taken once from the remaining time fed to the allocation above (a standing buffer
    // of about MoveOverheadSeconds maintained on the clock, costing each move only overhead/movesToGo).
    searchLimitToUse = AdjustedPerMoveSearchLimit(searchLimitToUse, paramsSearch,
                                                  applyMoveOverhead: !searchLimit.IsPerGameLimit);


    List<MGMove> searchMovesTablebaseRestricted = null;
    if (searchLimit.SearchMoves != null)
    {
      Position startPos = priorMoves.FinalPosition;
      foreach (Move move in searchLimit.SearchMoves)
      {
        searchMovesTablebaseRestricted.Add(MGMoveConverter.MGMoveFromPosAndMove(startPos, move));
      }
    }

    Manager = new(nnEvaluators,
                  paramsSearch, paramsSelect,
                  searchLimitToUse, limitManager, startTime,
                  gameMoveHistory, isFirstMoveOfGame,
                  forceNoTablebaseTerminals,
                  searchMovesTablebaseRestricted, priorMoves.FinalPosition.IsWhite,
                  fixedSearchLimit)
    {
      LastGameLimitInputs = gameLimitsInputs,
      LastGameLimitOutputs = gameLimitsOutputs
    };



    // Process graph rewrite if needed (handles memory pressure, low ratio triggers)
    Graph graphToUse = GraphReuseManager.PrepareGraphToUse(graphToReuse, searchRootNodeInfo, ref searchRootPathFromGraphRoot, paramsSearch, searchLimitToUse, priorMoves);

    // The graph stores were sized (and fixed) when the graph was created, based on the search limit
    // in effect at that time, and cannot subsequently grow. Therefore if this search is much larger
    // than the graph can accommodate (e.g. an interactive session issuing "go nodes 1" and then
    // "go nodes 100000") the stores would overflow partway through the search. In this situation
    // abandon the graph so a new one, sized for this search, is created below.
    // Note that the test is against actual remaining capacity (not the much larger allocation
    // GraphStoreSizeNodes would choose for a new graph, which anticipates reuse over many moves),
    // so an increase in search limit does not needlessly discard a graph which is still large enough.
    if (graphToUse != null)
    {
      int nodesInGraph = graphToUse.Store.NodesStore.NumTotalNodes;
      int estNodesThisSearch = searchLimitToUse.EstNumSearchNodes(nodesInGraph, 1 + (int)Manager.NNEvaluator0.EstNPSBatch);

      // The estimate above is explicitly not a hard upper bound, so require considerable headroom.
      const float SAFETY_MARGIN = 2.0f;
      long nodesNeeded = nodesInGraph + (long)(SAFETY_MARGIN * estNodesThisSearch) + 10_000;

      if (graphToUse.Store.MaxNodes < nodesNeeded)
      {
        GraphReuseManager.RecordAbandonStoreTooSmall();

        if (MCGSParamsFixed.GRAPH_REWRITE_DUMP_REUSE_DIAGNOSTICS)
        {
          ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
            $"Graph ABANDON: stores sized for {graphToUse.Store.MaxNodes:N0} nodes (currently {nodesInGraph:N0} used, "
          + $"{graphToUse.Store.FractionInUse:P0} full) are too small for search limit {searchLimitToUse} "
          + $"(needs about {nodesNeeded:N0}) {GraphReuseManager.StatsSummary()}");
        }

        graphToUse.Dispose();
        graphToUse = null;
        searchRootPathFromGraphRoot = null;
      }
    }

    // If a reusable graph was abandoned, the search will run from a fresh (empty) graph.
    // The allocation above used the (now-discarded) reuse candidate's root N, which with QuickMoves
    // enabled shrinks the time budget on the assumption of a warm tree
    // (see ManagerGameLimitCeresMCGS.MapFractionGraphReusedToShrinkageMultiplier). Recompute with a
    // cold-start input (root N = 0) so the from-scratch search is not starved of time. Guarded on
    // gameLimitsInputs.RootN (a plain int) rather than the now-disposed graph's node counts.
    if (graphToUse == null && gameLimitsInputs != null && gameLimitsInputs.RootN > 0)
    {
      SearchLimitType coldTargetType = searchLimit.IsTimeLimit ? SearchLimitType.SecondsPerMove
                                                               : SearchLimitType.NodesPerMove;
      ManagerGameLimitInputs coldInputs = new(priorMoves.FinalPosition,
                           paramsSearch, gameMoveHistory,
                           coldTargetType, 0, 0f,
                           ClockAfterMoveOverheadReserve(searchLimit, paramsSearch), searchLimit.ValueIncrement,
                           searchLimit.MaxTreeNodes, searchLimit.MaxTreeVisits,
                           float.NaN, float.NaN,
                           maxMovesToGo: searchLimit.MaxMovesToGo,
                           isFirstMoveOfGame: isFirstMoveOfGame, quickMoveEnabled: paramsSearch.EnableQuickMoves);

      // applyEarlySmoothing:false — the warm allocation already advanced the per-game early-smoothing
      // window for this move; don't double-count it on this recomputation.
      ManagerGameLimitOutputs coldOutputs = limitManager.ComputeMoveAllocation(coldInputs, applyEarlySmoothing: false);

      if (MCGSParamsFixed.GRAPH_REWRITE_DUMP_REUSE_DIAGNOSTICS)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
          $"Graph abandoned: recomputed cold-start move allocation {searchLimitToUse} -> {coldOutputs.LimitTarget} "
        + $"(was sized for warm root N={gameLimitsInputs.RootN:N0}).");
      }

      searchLimitToUse = AdjustedPerMoveSearchLimit(coldOutputs.LimitTarget, paramsSearch,
                                                    applyMoveOverhead: false);
      Manager.OverrideSearchLimit(searchLimitToUse, coldInputs, coldOutputs);
    }

    // Create new graph if needed (either no prior graph, or prior graph was abandoned)
    if (graphToUse == null)
    {
      int maxNodesInt = GraphStoreSizeNodes(searchLimit, fixedSearchLimit, paramsSearch);

      bool hasAction = Manager.NNEvaluator0.HasAction;

#if !ACTION_ENABLED
      // Without ACTION_ENABLED the per-child action buffer is never populated, so ActionHead FPU
      // would silently collapse to a flat constant. Fail loudly instead. Enable with -p:ActionEnabled=true.
      if (Manager.ParamsSelect.FPUMode == ParamsSelect.FPUType.ActionHead
       || Manager.ParamsSelect.FPUModeAtRoot == ParamsSelect.FPUType.ActionHead)
      {
        throw new Exception("FPUType.ActionHead (e.g. via UCI EnableActionHead) requires a build with ACTION_ENABLED (-p:ActionEnabled=true).");
      }
#endif

      if ((Manager.ParamsSelect.FPUMode == ParamsSelect.FPUType.ActionHead
        || Manager.ParamsSelect.FPUModeAtRoot == ParamsSelect.FPUType.ActionHead)
       && !hasAction)
      {
        throw new Exception("FPUType.ActionHead requires a neural network with an action head output.");
      }

      // Right-size the transposition dictionaries to the realistic expected number of distinct
      // positions for this search budget, instead of the worst-case node-buffer reservation (maxNodes).
      int dictionarySizeHint = EstimateInitialDictionaryCapacity(searchLimitToUse, Manager.ParamsSearch, 1 + (int)Manager.NNEvaluator0.EstNPSBatch);
      //      ConsoleUtils.WriteLineColored(ConsoleColor.Red, $"dict= {(float)dictionarySizeHint/1_000_000}  graph={maxNodesInt/1_000_000.0}");

      graphToUse = new(maxNodesInt, hasAction,
                       Manager.ParamsSearch.EnableState,
                       Manager.ParamsSearch.EnableGraph,
                       Manager.ParamsSearch.PathTranspositionMode == PathMode.PositionEquivalence,
                       MCGSParamsFixed.TryEnableLargePages,
                       Manager.ParamsSearch.EnablePseudoTranspositionBlending,
                       priorMoves,
                       Manager.ParamsSearch.TestFlag,
                       Manager.ParamsSearch.MaintainSiblingSets,
                       dictionarySizeHint);

      // Re-establish the cross-graph evaluation-reuse provider that was attached to the abandoned graph
      // (the one-shot PossiblyInitializeForOpponentGraphReuse will not re-run for this new graph).
      graphToUse.ReuseGraphProvider = priorReuseGraphProvider;
    }

    Manager.Engine = new MCGSEngine(Manager, selectWorkerPools, graphToUse, searchRootPathFromGraphRoot == null ? []
                                                                                              : [.. searchRootPathFromGraphRoot]);

#if DEBUG
    // Verify the search root actually represents the current position before searching it.
    MCGSRootConsistencyCheck.Validate(Manager.Engine, priorMoves.FinalPosition, "search start", out _);
#endif    

    StartSearchN = SearchRootNode.N;
    (MGMove bestMove, BestMoveInfoMCGS moveInfo) = DoSearch(Manager, verbose, progressCallback,
                                                            moveImmediateIfOnlyOneMove, forcedMove);
    BestMove = bestMove;
  }


  /// <summary>
  /// Returns the number of nodes for which the stores of a new graph should be sized
  /// to accommodate a search with the specified limit (allowing also for the graph
  /// possibly being reused over many subsequent moves).
  /// </summary>
  /// <param name="searchLimit">Search limit for which the graph is being sized.</param>
  /// <param name="fixedSearchLimit">Optionally a fixed (per-move) limit known to apply to all moves.</param>
  /// <param name="paramsSearch">Search parameters (used to read graph reuse and memory settings).</param>
  internal static int GraphStoreSizeNodes(SearchLimit searchLimit, SearchLimit fixedSearchLimit, ParamsSearch paramsSearch)
  {
    const long MAX_NODES = 1_100_000_000L;

    // Attempt to find some safe (hopefully lower value)
    // to use for max nodes than MAX_NODES to reduce virtual memory reservation length.
    // Note that if GraphReuseRewriteEnabled then we can be more aggressive in allowing more nodes
    // because we will rewrite the graph to reduce if approaches the limit.
    // TODO: possibly unify this with code already in SearchLimit
    long MAX_MOVES_PER_GRAPH = paramsSearch.GraphReuseEnabled ? (paramsSearch.GraphReuseRewriteEnabled ? 150L : 400L) : 5L;
    const long MAX_NODES_PER_SECOND = 500_000L;
    long maxNodes = searchLimit.Type switch
    {
      SearchLimitType.BestValueMove => 1,
      SearchLimitType.BestActionMove => 1,

      SearchLimitType.NodesPerMove => (long)(MAX_MOVES_PER_GRAPH * (float)(searchLimit.Value + searchLimit.ValueIncrement)),
      SearchLimitType.NodesForAllMoves => (long)(searchLimit.Value + searchLimit.ValueIncrement * MAX_MOVES_PER_GRAPH),

      SearchLimitType.SecondsPerMove => (long)((searchLimit.Value + searchLimit.ValueIncrement) * MAX_NODES_PER_SECOND * MAX_MOVES_PER_GRAPH),
      SearchLimitType.SecondsForAllMoves => (long)((searchLimit.Value + (searchLimit.ValueIncrement * MAX_MOVES_PER_GRAPH)) * MAX_NODES_PER_SECOND),
      _ => MAX_NODES
    };

    maxNodes = Math.Min(MAX_NODES, maxNodes);

    // Possibly apply tighter constraint based on fixedSearchLimit
    if (fixedSearchLimit != null && fixedSearchLimit.IsNodesLimit)
    {
      long fixedMax = MAX_MOVES_PER_GRAPH * (long)(fixedSearchLimit.Value + fixedSearchLimit.ValueIncrement);
      maxNodes = Math.Min(maxNodes, fixedMax);
    }

    // Possibly apply constraint based on max memory
    long maxBytes = paramsSearch.MaxMemoryBytes;
    const long MIN_BYTES_PER_NODE = 200; // average probably more typically 400 considering all associated data structures
    long maxNodesAllowedInMemory = maxBytes / MIN_BYTES_PER_NODE;
    maxNodes = Math.Min(maxNodes, maxNodesAllowedInMemory);

    return (int)Math.Min(maxNodes + 1000, MAX_NODES);
  }


  /// <summary>
  /// Estimates a sensible initial capacity (in entries) for the per-graph transposition dictionaries,
  /// based on the search budget. This lets the dictionary be sized up front to roughly the number of
  /// distinct positions it will hold, avoiding a cascade of incremental resizes (directory doublings and
  /// bucket pre-allocation with the extendible map, or stop-the-world rehashes under the legacy
  /// ConcurrentDictionary) as a long search grows the table from a small default to tens of millions of
  /// entries while holding locks (a stall that idles the GPU).
  ///
  /// A tiny budget (e.g. a single node) yields a tiny hint; a large budget (e.g. a long tournament time
  /// control arriving via UCI) yields a larger one. The returned value is clamped here and again,
  /// defensively, in Graph.Initialize.
  /// </summary>
  /// <param name="limit">The per-move search limit actually being used (per-game limits have already
  /// been mapped to a per-move target by the limits manager).</param>
  /// <param name="paramsSearch">Search parameters (used to read graph-reuse settings).</param>
  /// <param name="estimatedEPS">Estimated maximum evaluations per second from this backend.</param>
  static int EstimateInitialDictionaryCapacity(SearchLimit limit, ParamsSearch paramsSearch, int estimatedEPS)
  {

    // Expected nodes added by this move's search (fresh graph => 0 initial nodes).
    long perMoveNodes = Math.Max(1, limit.EstNumSearchNodes(0, estimatedEPS));

    // The graph (and its dictionary) persists across the multiple moves that share a reused graph, so
    // anticipate some cross-move accumulation of distinct positions. Without reuse a single search adds
    // only modestly more positions than its final tree size.
    double accumFactor = paramsSearch.GraphReuseEnabled
                           ? MCGSParamsFixed.DICTIONARY_SIZE_HINT_REUSE_ACCUM_FACTOR
                           : 1.25;

    long estEntries = (long)(perMoveNodes * accumFactor);

    return (int)Math.Clamp(estEntries, MCGSParamsFixed.DICTIONARY_SIZE_HINT_MIN, MCGSParamsFixed.DICTIONARY_SIZE_HINT_MAX);
  }


  /// <summary>
  /// Returns the specified per-move search limit adjusted for actual execution:
  ///
  ///   - if applyMoveOverhead, the configured move overhead (ParamsSearch.MoveOverheadSeconds,
  ///     settable via the UCI MoveOverheadMs option) is held back from a time limit, so that the
  ///     time spent emitting the move and any GUI/network latency do not push the move past its
  ///     allotted time. The reserve is capped at half of the allotted time so that a short search
  ///     (e.g. a "go movetime 100" analysis probe, where the overhead exceeds the whole budget) is
  ///     shortened proportionally rather than collapsed to nothing.
  ///     This applies only to true per-move limits; a limit allocated from a per-game clock has the
  ///     overhead reserved once on the clock instead (see ClockAfterMoveOverheadReserve).
  ///
  ///   - any MaxTreeNodes configured in Ceres.json is installed if the limit does not already
  ///     carry an explicit value (the UCI layer applies this setting itself; this covers the
  ///     in-process entry points, such as tournaments and test suites, which do not).
  ///     Note that the store's own capacity is enforced independently of this setting
  ///     (see MCGSManager.CalcSearchStopStatus).
  /// </summary>
  /// <param name="limit">The per-move limit about to be searched.</param>
  /// <param name="paramsSearch"></param>
  /// <param name="applyMoveOverhead">If the move overhead should be held back from this limit
  /// (false when the limit came from a per-game clock which was already reserved against).</param>
  /// <returns></returns>
  static SearchLimit AdjustedPerMoveSearchLimit(SearchLimit limit, ParamsSearch paramsSearch,
                                                bool applyMoveOverhead)
  {
    if (limit.MaxTreeNodes is null && CeresUserSettingsManager.Settings.MaxTreeNodes is not null)
    {
      limit = limit with { MaxTreeNodes = CeresUserSettingsManager.Settings.MaxTreeNodes };
    }

    if (!applyMoveOverhead || !limit.IsTimeLimit || paramsSearch.MoveOverheadSeconds <= 0 || limit.Value <= 0)
    {
      return limit;
    }

    const float MIN_SEARCH_SECONDS = 0.01f;
    const float MAX_FRACTION_RESERVED = 0.5f;

    float overhead = Math.Min(paramsSearch.MoveOverheadSeconds, MAX_FRACTION_RESERVED * limit.Value);
    return limit with { Value = Math.Max(MIN_SEARCH_SECONDS, limit.Value - overhead) };
  }


  /// <summary>
  /// Returns the remaining game clock time of the specified per-game limit less the configured
  /// move overhead (ParamsSearch.MoveOverheadSeconds, settable via the UCI MoveOverheadMs option).
  ///
  /// Reserving once from the clock (rather than from every per-move allocation) maintains a
  /// standing buffer of about MoveOverheadSeconds against move emission and GUI/network latency.
  /// </summary>
  /// <param name="searchLimit">A per-game limit whose Value is the remaining clock time.</param>
  /// <param name="paramsSearch"></param>
  /// <returns></returns>
  static float ClockAfterMoveOverheadReserve(SearchLimit searchLimit, ParamsSearch paramsSearch)
  {
    if (!searchLimit.IsTimeLimit || paramsSearch.MoveOverheadSeconds <= 0 || searchLimit.Value <= 0)
    {
      return searchLimit.Value;
    }

    const float MAX_FRACTION_RESERVED = 0.5f;

    float reserve = Math.Min(paramsSearch.MoveOverheadSeconds, MAX_FRACTION_RESERVED * searchLimit.Value);
    return searchLimit.Value - reserve;
  }
}
