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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

using Ceres.Chess.Probing;
using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Search.ProbeGraft;

/// <summary>
/// Per-search coordinator of the PICKET M1 refutation-grafting feature: owns the resolved
/// probe source session, the result inbox, the target ledger (ProbeGraftTargeter), the graft
/// scheduler (ProbeGraftScheduler), and the per-search statistics (ProbeGraftStats).
///
/// Created by MCGSManager's constructor when ParamsProbeGraft.Mode != Disabled (an
/// unknown ProbeSourceID fails fast there); ended by MCGSManager.DoSearchInner at end of
/// search; disposed with the manager as a safety net. An MCGSManager is strictly
/// one-search, so this object is too.
///
/// Threading and invariants:
///  - Probe completion callbacks arrive on arbitrary threads and ONLY enqueue into the
///    inbox (plus an interlocked pending count); they never touch the graph, params, or
///    any other stats field.
///  - All graph reads/writes, all stats mutation, and all RunProbeSpecs calls happen on
///    the pump thread only, at the post-batch quiescent point (PumpAtQuiescentPoint is
///    invoked from MCGSIterator.RunOnce at the same site as MCGSEngine.PostBatchHook,
///    after backup has completed and all coordinator gates are exited).
///  - The RunProbeSpecs contract (verified in MCGSIterator.Probe.cs): quiescent graph
///    required (it self-checks root in-flight and throws), single-iterator harnesses only
///    (DualOverlappedIterators is validated off), commitInsteadOfDrop = true for grafts,
///    per-call spec visit totals chunked to Execution.MaxBatchSize.
///  - End of search: EndSession is called before the coordinator is dropped; the
///    IProbeSource contract guarantees no callbacks after it returns, so there is
///    no use-after-dispose of the inbox.
///  - Feature exceptions never propagate into search: the pump boundary catch logs the
///    first exception once and disables the feature for the remainder of the search
///    (search integrity beats feature integrity).
/// </summary>
public sealed class ProbeGraftCoordinator : IDisposable
{
  /// <summary>|delta Q| threshold used by impact stats and Shadow verdicts
  /// (matches BottomUpQRecalculator.LARGE_DELTA_THRESHOLD).</summary>
  const double LARGE_DELTA_THRESHOLD = 0.03;

  /// <summary>Maximum retained per-event log lines (VerboseEventLogging).</summary>
  const int MAX_EVENT_LOG_LINES = 512;

  /// <summary>Process-wide monotonically increasing probe request ID.</summary>
  static long nextRequestID;

  /// <summary>One recorded would-have-grafted event (Shadow mode) awaiting its verdict.</summary>
  sealed class ShadowEvent
  {
    public int AnchorNodeIndex;
    public bool IsRefutation;
    public int DominantChildIndexAtTrigger;
    public double DominantEdgeQAtTrigger;
    public double AnchorQAtTrigger;
    public MGMove ProbeBestMove;
  }

  /// <summary>The owning (one-search) manager.</summary>
  public readonly MCGSManager Manager;

  /// <summary>Per-search statistics (exposed as MCGSManager.ProbeGraftStats).</summary>
  public readonly ProbeGraftStats Stats = new();

  readonly ParamsProbeGraft Params;
  readonly IProbeSource source;

  /// <summary>This search's session on the (possibly process-wide shared) probe source.</summary>
  IProbeSession session;
  readonly ProbeGraftTargeter targeter;
  readonly ProbeGraftScheduler scheduler;
  readonly CancellationTokenSource cts = new();

  /// <summary>Result inbox: the completion callback only enqueues here.</summary>
  readonly ConcurrentQueue<ProbeResult> inbox = new();

  /// <summary>Interlocked count of enqueued-but-undrained results (cheap idle check).</summary>
  int numPendingResults;

  /// <summary>Paces target sweeps (restarted after each sweep).</summary>
  readonly Stopwatch sweepTimer = Stopwatch.StartNew();

  readonly List<ShadowEvent> shadowEvents = new();
  readonly List<string> eventLog = new();

  bool disabledByError;
  bool endSearchDone;
  bool disposed;


  /// <summary>
  /// Constructor: validates parameters, resolves the probe source from the registry
  /// (unknown ID throws - fail fast at search start, never silently), and begins the
  /// probe session.
  /// </summary>
  /// <param name="manager"></param>
  public ProbeGraftCoordinator(MCGSManager manager)
  {
    Manager = manager;
    Params = manager.ParamsSearch.ProbeGraft;

    Debug.Assert(Params != null && Params.Mode != ParamsProbeGraft.ModeType.Disabled);
    Params.Validate(manager.ParamsSearch);

    if (!ProbeSourceRegistry.TryCreate(Params.ProbeSourceID, out source))
    {
      throw new Exception($"ParamsProbeGraft.ProbeSourceID '{Params.ProbeSourceID}' "
                        + "is not registered with ProbeSourceRegistry.");
    }

    targeter = new ProbeGraftTargeter(Params);
    scheduler = new ProbeGraftScheduler(this, Params, Stats);

    session = source.BeginSession(OnProbeCompleted);
  }


  /// <summary>
  /// Completion sink handed to the probe source (arbitrary thread): enqueue-only.
  /// </summary>
  /// <param name="result"></param>
  void OnProbeCompleted(ProbeResult result)
  {
    inbox.Enqueue(result);
    Interlocked.Increment(ref numPendingResults);
  }


  /// <summary>
  /// Single entry point from search, invoked at the end of each MCGSIterator.RunOnce batch
  /// (backup complete, coordinator gates exited, graph quiescent in single-iterator
  /// harnesses). Cheap when idle: a pending-count read, a scheduler check and a timestamp
  /// comparison. Body order: (1) drain inbox and interpret results; (2) advance active
  /// grafts one ply; (3) run the target sweep when its interval elapsed.
  /// </summary>
  /// <param name="iterator"></param>
  public void PumpAtQuiescentPoint(MCGSIterator iterator)
  {
    if (disabledByError || endSearchDone)
    {
      return;
    }

    bool anyResults = Volatile.Read(ref numPendingResults) != 0;
    bool sweepDue = sweepTimer.Elapsed.TotalSeconds >= Params.TargetSweepIntervalSeconds;
    bool rootAdaptDue = Params.RootAdaptMode == ParamsProbeGraft.RootAdaptModeType.Active && !rootAdaptDone;
    if (!anyResults && !scheduler.HasWork && !sweepDue && !rootAdaptDue)
    {
      return;
    }

    long t0 = Stopwatch.GetTimestamp();
    try
    {
      // Once the search has decided to stop, stop consuming budget (results still drain
      // so the stats and Shadow ledgers stay complete).
      bool searchStopping = Manager.StopStatus != MCGSManager.SearchStopStatus.Continue;

      if (anyResults)
      {
        long tDrain = Stopwatch.GetTimestamp();
        DrainInbox(iterator.Engine);
        Stats.TimeDrainMs += MsSince(tDrain);
      }

      if (!searchStopping && Params.Mode == ParamsProbeGraft.ModeType.Active && scheduler.HasWork)
      {
        long tGraft = Stopwatch.GetTimestamp();
        ushort rootTopBefore = TopNRootMoveRaw(iterator.Engine);
        bool applied = scheduler.AdvanceOnePly(iterator);
        if (applied && TopNRootMoveRaw(iterator.Engine) != rootTopBefore)
        {
          Stats.NumRootBestMoveChangesAfterGraft++;
        }
        Stats.TimeGraftAdvanceMs += MsSince(tGraft);
      }

      if (!searchStopping && sweepDue)
      {
        long tSweep = Stopwatch.GetTimestamp();
        RunSweepAndSubmit(iterator.Engine);
        sweepTimer.Restart();
        Stats.TimeTargetSweepMs += MsSince(tSweep);
      }

      if (rootAdaptDue)
      {
        if (searchStopping)
        {
          FinishRootAdapt(iterator.Engine, abandoned: true);
        }
        else
        {
          PumpRootAdapt(iterator);
        }
      }
    }
    catch (Exception exc)
    {
      DisableAfterError(exc);
    }
    finally
    {
      Stats.TimePumpTotalMs += MsSince(t0);
    }
  }


  /// <summary>
  /// Finalizes the feature for this search: cancels/ends the probe session (no callbacks
  /// occur after EndSession returns), discards any late results, computes impact deltas
  /// and Shadow verdicts against the final graph, and prints the one-line summary when
  /// enabled. Idempotent; invoked from MCGSManager.DoSearchInner at end of search.
  /// </summary>
  public void EndSearch()
  {
    if (!endSearchDone && Params.RootAdaptMode == ParamsProbeGraft.RootAdaptModeType.Active)
    {
      try
      {
        if (rootAdaptActive)
        {
          FinishRootAdapt(Manager.Engine, abandoned: true);
        }
        if (rootAdaptFlipped && rootAdaptMoveRaw == TopNRootMoveRaw(Manager.Engine))
        {
          Stats.NumRootAdaptFlipsHeld++;
        }
      }
      catch (Exception exc)
      {
        DisableAfterError(exc);
      }
    }

    if (endSearchDone)
    {
      return;
    }
    endSearchDone = true;

    ShutdownSession();

    // Like the pump, end-of-search work must never propagate an exception into search.
    try
    {
      while (inbox.TryDequeue(out _))
      {
        Interlocked.Decrement(ref numPendingResults);
        Stats.NumResultsDiscardedAtEnd++;
      }

      Stats.NumDistinctNodesProbed = targeter.NumDistinctNodesProbed;

      if (Manager.Engine != null)
      {
        ComputeImpactStats(Manager.Engine.Graph);
        ComputeShadowVerdicts(Manager.Engine.Graph);
      }

      // ParamsSearch.DumpHybridSearchStats is the research switch and takes precedence over the
      // terse per-search line; Off leaves the legacy EnableStatsSummary behavior untouched.
      ParamsSearch.HybridStatsDumpType dumpMode = Manager.ParamsSearch.DumpHybridSearchStats;
      if (dumpMode == ParamsSearch.HybridStatsDumpType.Full)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Cyan,
          Stats.SearchReport(Params.Mode, Manager.NumEvalsThisSearch, source?.Description));
      }
      else if (dumpMode == ParamsSearch.HybridStatsDumpType.Compact || Params.EnableStatsSummary)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Cyan,
                                      Stats.SummaryLine(Params.Mode, Manager.NumEvalsThisSearch));
      }

      // Optionally fold this search into the process-wide aggregate used by multi-search
      // harnesses (measurement campaigns, matches, tournament managers).
      if (ProbeGraftStats.EnableGlobalAccumulation)
      {
        ProbeGraftStats.AccumulateGlobal(Stats, Params.Mode, Manager.NumEvalsThisSearch);
      }
    }
    catch (Exception exc)
    {
      DisableAfterError(exc);
    }
  }


  /// <summary>
  /// Safety net: ensures the probe session is ended even if the search never completed
  /// normally. The source instance itself may be shared (registry factory) and is never
  /// disposed here.
  /// </summary>
  public void Dispose()
  {
    if (disposed)
    {
      return;
    }
    disposed = true;

    if (!endSearchDone)
    {
      endSearchDone = true;
      ShutdownSession();
    }

    cts.Dispose();
  }


  /// <summary>Retained per-event log lines (populated when VerboseEventLogging).</summary>
  public IReadOnlyList<string> EventLog => eventLog;


  #region Result drain and trigger interpretation

  /// <summary>
  /// Verdict of the most recent completed root probe in this search: what would play at
  /// the root, and by how much it preferred that over the move the search was favoring when the
  /// probe was issued. Consumed at move commit (Shadow: recorded only).
  /// </summary>
  public readonly struct RootVerdict
  {
    public readonly bool HasValue;
    public readonly MGMove ProbeBestMove;
    public readonly MGMove SearchDominantMove;
    public readonly int MarginCp;
    public readonly int ProbeDepth;

    public RootVerdict(MGMove probeBestMove, MGMove searchDominantMove, int marginCp, int probeDepth)
    {
      HasValue = true;
      ProbeBestMove = probeBestMove;
      SearchDominantMove = searchDominantMove;
      MarginCp = marginCp;
      ProbeDepth = probeDepth;
    }
  }


  /// <summary>Most recent root probe verdict of this search (see RootVerdict).</summary>
  public RootVerdict LastRootVerdict { get; private set; }


  /// <summary>
  /// Records the root probe verdict. A verdict only counts as a veto candidate when the probe both
  /// searched deeply enough to be trusted and preferred a different move by a clear margin; the
  /// comparison against the move actually played happens at commit, since the search continues
  /// after this probe returns and may change its mind on its own.
  /// </summary>
  /// <param name="snapshot"></param>
  /// <param name="result"></param>
  void RecordRootVerdict(ProbeGraftTargetSnapshot snapshot, ProbeResult result)
  {
    Stats.NumRootProbeVerdicts++;

    if (result.Depth < Params.RootVetoMinProbeDepth || result.DominantMoveScoreCp == int.MinValue)
    {
      return;
    }

    int marginCp = result.BestScoreCp - result.DominantMoveScoreCp;
    LastRootVerdict = new RootVerdict(result.BestMove, snapshot.DominantMove, marginCp, result.Depth);

    if (result.BestMove != snapshot.DominantMove && marginCp >= Params.RootVetoMinMarginCp)
    {
      Stats.NumRootVetoCandidates++;
    }
  }


  void DrainInbox(MCGSEngine engine)
  {
    while (inbox.TryDequeue(out ProbeResult result))
    {
      Interlocked.Decrement(ref numPendingResults);
      InterpretResult(engine, result);
    }
  }


  /// <summary>
  /// Applies the trigger rules to one completed probe result: staleness check, then
  /// refutation trigger (primary), discovery trigger (secondary), else agreement.
  /// Ambiguous snapshots are discarded and counted - a discarded probe costs nothing.
  /// </summary>
  void InterpretResult(MCGSEngine engine, ProbeResult result)
  {
    ProbeGraftTargetSnapshot snapshot = (ProbeGraftTargetSnapshot)result.RequesterToken;

    if (!result.Completed)
    {
      Stats.NumProbesIncomplete++;
      return;
    }

    Stats.NumProbesCompleted++;
    Stats.ProbeLatencySumMs += result.ElapsedMilliseconds;
    if (result.ElapsedMilliseconds > Stats.ProbeLatencyMaxMs)
    {
      Stats.ProbeLatencyMaxMs = result.ElapsedMilliseconds;
    }
    Stats.ProbeNodesSum += result.Nodes;
    Stats.ProbeDepthSum += result.Depth;

    // Staleness: within a search node indices are stable (graph rewrites happen between
    // searches), but the node must still be eligible and, for refutation triggering, the
    // dominant child must still be the snapshot's.
    GNode node = TryResolveNode(engine.Graph, snapshot.NodeIndex);
    bool eligible = !node.IsNull
                 && node.IsEvaluated
                 && node.N > 0
                 && !node.Terminal.IsTerminal()
                 && node.NumPolicyMoves > 0
                 && (!node.IsSearchRoot || Params.TargetIncludeSearchRoot);
    if (!eligible)
    {
      Stats.NumResultsStale++;
      return;
    }

    // The dominant child must still be the snapshot's dominant child; when the visit
    // distribution has shifted since the probe was issued, the result answers a question
    // the graph is no longer asking - discard it (a discarded probe costs nothing).
    if (snapshot.DominantChildIndex >= 0
     && DominantChildIndex(node) != snapshot.DominantChildIndex)
    {
      Stats.NumResultsStale++;
      return;
    }

    if (Params.RootVetoMode != ParamsProbeGraft.RootVetoModeType.Disabled && node.IsSearchRoot)
    {
      RecordRootVerdict(snapshot, result);
    }

    bool refutation = result.DominantMoveScoreCp != int.MinValue
                   && snapshot.DominantChildIndex >= 0
                   && result.BestMove != snapshot.DominantMove
                   && (result.BestScoreCp - result.DominantMoveScoreCp) >= Params.TriggerMarginCp
                   && result.Depth >= Params.TriggerMinProbeDepth
                   && snapshot.DominantFractionGateHeld;

    if (refutation)
    {
      Stats.NumTriggersRefutation++;
      Stats.RecordTriggerDepth(snapshot.Depth);
      MGMove[] moves = BuildRefutationLine(snapshot.DominantMove, result.RefutationPV);
      HandleTrigger(node, snapshot, result, moves, isRefutation: true,
                    marginCp: result.BestScoreCp - result.DominantMoveScoreCp);
      return;
    }

    // Discovery trigger: the prober likes a move the policy effectively ignores.
    if (Params.PriorNudgeFloor > 0
     && result.Depth >= Params.TriggerMinProbeDepth
     && result.PV.Count > 0
     && result.BestMove != snapshot.DominantMove)
    {
      float bestMovePrior = PriorOfMoveAtNode(node, result.BestMove);
      if (!float.IsNaN(bestMovePrior) && bestMovePrior < Params.PriorNudgeFloor)
      {
        Stats.NumTriggersDiscovery++;
        Stats.RecordTriggerDepth(snapshot.Depth);
        MGMove[] moves = TruncateLine(result.PV);
        HandleTrigger(node, snapshot, result, moves, isRefutation: false, marginCp: 0);
        return;
      }
    }

    // Agreement - the expected common case; its cheapness is the point.
    Stats.NumProbesAgreed++;
  }


  /// <summary>
  /// Routes a triggered event: Active mode queues a graft plan; Shadow mode records the
  /// would-have-grafted event for its end-of-search verdict.
  /// </summary>
  void HandleTrigger(GNode node, ProbeGraftTargetSnapshot snapshot, ProbeResult result,
                     MGMove[] moves, bool isRefutation, int marginCp)
  {
    if (Params.VerboseEventLogging)
    {
      LogEvent($"trigger {(isRefutation ? "ref" : "disc")} node=#{snapshot.NodeIndex} N={snapshot.N} "
             + $"dom={snapshot.DominantMove} best={result.BestMove} marginCp={marginCp} depth={result.Depth}");
    }

    if (moves.Length == 0)
    {
      return;
    }

    if (Params.Mode == ParamsProbeGraft.ModeType.Active)
    {
      scheduler.Enqueue(new ProbeGraftPlan()
      {
        AnchorNodeIndex = snapshot.NodeIndex,
        Moves = moves,
        IsRefutation = isRefutation,
        MarginCp = marginCp,
        ProbeDepth = result.Depth,
      });
    }
    else
    {
      double dominantEdgeQ = snapshot.DominantChildIndex >= 0
                          && snapshot.DominantChildIndex < node.NumEdgesExpanded
                           ? node.ChildEdgeAtIndex(snapshot.DominantChildIndex).Q
                           : double.NaN;
      shadowEvents.Add(new ShadowEvent()
      {
        AnchorNodeIndex = snapshot.NodeIndex,
        IsRefutation = isRefutation,
        DominantChildIndexAtTrigger = snapshot.DominantChildIndex,
        DominantEdgeQAtTrigger = dominantEdgeQ,
        AnchorQAtTrigger = node.Q,
        ProbeBestMove = result.BestMove,
      });
    }
  }


  MGMove[] BuildRefutationLine(MGMove dominantMove, IReadOnlyList<MGMove> refutationPV)
  {
    int numReply = Math.Min(refutationPV.Count, Params.GraftMaxPlies - 1);
    MGMove[] moves = new MGMove[1 + numReply];
    moves[0] = dominantMove;
    for (int i = 0; i < numReply; i++)
    {
      moves[1 + i] = refutationPV[i];
    }
    return moves;
  }


  MGMove[] TruncateLine(IReadOnlyList<MGMove> pv)
  {
    int num = Math.Min(pv.Count, Params.GraftMaxPlies);
    MGMove[] moves = new MGMove[num];
    for (int i = 0; i < num; i++)
    {
      moves[i] = pv[i];
    }
    return moves;
  }

  #endregion

  #region Target sweep and probe submission

  PositionWithHistory searchRootPositionWithHistory;


  /// <summary>
  /// Returns the position history of the SEARCH root (cached; fixed for the duration of a search).
  ///
  /// Manager.StartPosAndPriorMoves is the history of the GRAPH root, which is the same node only
  /// when the graph is not reused. Under graph reuse (as in game play) the search root sits below
  /// the graph root, so the moves along Engine.SearchRootPathFromGraphRoot must be appended before
  /// the sweep's root-relative move path (otherwise those moves are illegal in the resulting
  /// position and probe submission throws).
  /// </summary>
  /// <returns></returns>
  PositionWithHistory SearchRootPositionWithHistory()
  {
    if (searchRootPositionWithHistory == null)
    {
      PositionWithHistory position = new PositionWithHistory(Manager.StartPosAndPriorMoves);
      GraphRootToSearchRootNodeInfo[] pathFromGraphRoot = Manager.Engine.SearchRootPathFromGraphRoot;
      if (pathFromGraphRoot != null)
      {
        foreach (GraphRootToSearchRootNodeInfo nodeInfo in pathFromGraphRoot)
        {
          position.AppendMove(nodeInfo.MoveToChild);
        }
      }
      searchRootPositionWithHistory = position;
    }

    return searchRootPositionWithHistory;
  }


  /// <summary>
  /// Runs one target sweep and submits probe requests for the selected snapshots, up to
  /// the in-flight cap. Request positions are the search-root history plus the walk's move
  /// path (preserving repetition/50-move context for the prober).
  /// </summary>
  void RunSweepAndSubmit(MCGSEngine engine)
  {
    Stats.NumTargetSweeps++;
    List<ProbeGraftTargetSnapshot> selected = targeter.Sweep(engine);
    Stats.NumTargetsSelected += selected.Count;

    int capacity = Params.MaxInFlightProbes - session.NumInFlight;
    foreach (ProbeGraftTargetSnapshot snapshot in selected)
    {
      if (capacity <= 0)
      {
        break;
      }

      PositionWithHistory position = new PositionWithHistory(SearchRootPositionWithHistory());
      foreach (MGMove move in snapshot.MovesFromRoot)
      {
        position.AppendMove(move);
      }

      ProbeRequest request = new()
      {
        RequestID = Interlocked.Increment(ref nextRequestID),
        RequesterToken = snapshot,
        Position = position,
        DominantMove = snapshot.DominantMove,
        NodeBudget = Params.ProbeNodeBudget,
        CancellationToken = cts.Token,
      };

      if (session.SubmitProbe(request))
      {
        Stats.NumProbesSubmitted++;
        Stats.RecordProbeDepth(snapshot.Depth);
        targeter.RecordProbeSubmitted(snapshot);
        capacity--;
      }
      else
      {
        Stats.NumProbesRejectedBySource++;
      }
    }
  }

  #endregion

  #region End-of-search impact and Shadow verdicts

  void ComputeImpactStats(Graph graph)
  {
    foreach ((int anchorIndex, double qBefore) in scheduler.CompletedGraftImpacts)
    {
      GNode anchor = TryResolveNode(graph, anchorIndex);
      if (anchor.IsNull)
      {
        continue;
      }

      double delta = Math.Abs(anchor.Q - qBefore);
      Stats.SumAbsAnchorQDeltaAtEnd += delta;
      if (delta > Stats.MaxAbsAnchorQDeltaAtEnd)
      {
        Stats.MaxAbsAnchorQDeltaAtEnd = delta;
      }
      if (delta > LARGE_DELTA_THRESHOLD)
      {
        Stats.NumAnchorsQMovedOverThreshold++;
      }

      LogEvent($"impact anchor=#{anchorIndex} qBefore={qBefore:F3} qEnd={anchor.Q:F3} absDQ={delta:F3}");
    }
  }


  /// <summary>
  /// Computes the per-event Shadow verdicts against the final graph: an event is
  /// CONFIRMED when plain search resolved it on its own - for a refutation event, the
  /// anchor's visit-dominant child changed away from the refuted move or the refuted
  /// edge's (child-perspective) Q rose by more than the threshold (the move was revealed
  /// worse for the anchor); for a discovery event, the probe's move became visit-dominant
  /// or the anchor Q rose by more than the threshold. Everything else is UNRESOLVED -
  /// the cases where Active mode would differ from baseline.
  /// </summary>
  void ComputeShadowVerdicts(Graph graph)
  {
    Stats.ShadowNumEvents = shadowEvents.Count;

    foreach (ShadowEvent shadowEvent in shadowEvents)
    {
      bool confirmed = false;
      GNode node = TryResolveNode(graph, shadowEvent.AnchorNodeIndex);
      if (!node.IsNull && node.NumEdgesExpanded > 0)
      {
        int dominantNow = DominantChildIndex(node);

        if (shadowEvent.IsRefutation)
        {
          bool dominantChanged = dominantNow != shadowEvent.DominantChildIndexAtTrigger;
          bool refutedEdgeQRose = !double.IsNaN(shadowEvent.DominantEdgeQAtTrigger)
                               && shadowEvent.DominantChildIndexAtTrigger < node.NumEdgesExpanded
                               && (node.ChildEdgeAtIndex(shadowEvent.DominantChildIndexAtTrigger).Q
                                   - shadowEvent.DominantEdgeQAtTrigger) > LARGE_DELTA_THRESHOLD;
          confirmed = dominantChanged || refutedEdgeQRose;
        }
        else
        {
          bool becameDominant = false;
          if (dominantNow >= 0)
          {
            MGPosition nodePos = node.CalcPosition();
            becameDominant = node.ChildEdgeAtIndex(dominantNow).MoveMGFromPos(in nodePos)
                          == shadowEvent.ProbeBestMove;
          }
          bool anchorQRose = (node.Q - shadowEvent.AnchorQAtTrigger) > LARGE_DELTA_THRESHOLD;
          confirmed = becameDominant || anchorQRose;
        }
      }

      if (confirmed)
      {
        Stats.ShadowNumConfirmedBySearch++;
      }
      else
      {
        Stats.ShadowNumUnresolved++;
      }
    }
  }

  #endregion

  #region Helpers

  /// <summary>
  /// Resolves a node index against the graph, returning a null GNode when out of range.
  /// </summary>
  /// <param name="graph"></param>
  /// <param name="nodeIndex"></param>
  internal static GNode TryResolveNode(Graph graph, int nodeIndex)
  {
    int numUsed = graph.Store.NodesStore.NumUsedNodes;
    if (nodeIndex < GNodeStore.FIRST_ALLOCATED_INDEX
     || nodeIndex >= GNodeStore.FIRST_ALLOCATED_INDEX + numUsed)
    {
      return default;
    }

    return graph[nodeIndex];
  }


  /// <summary>
  /// Index of the node's most-visited expanded child edge (-1 when no child has visits).
  /// </summary>
  /// <param name="node"></param>
  static int DominantChildIndex(GNode node)
  {
    int dominantIndex = -1;
    int dominantN = 0;
    int numExpanded = node.NumEdgesExpanded;
    for (int slot = 0; slot < numExpanded; slot++)
    {
      GEdge edge = node.ChildEdgeAtIndex(slot);
      if (edge.N > dominantN)
      {
        dominantN = edge.N;
        dominantIndex = slot;
      }
    }
    return dominantIndex;
  }


  /// <summary>
  /// Policy prior of the given move at the node (expanded edge or unexpanded header),
  /// NaN when the move is absent from the node's policy move list.
  /// </summary>
  static float PriorOfMoveAtNode(GNode node, MGMove move)
  {
    if (node.IsPendingPolicyCopy)
    {
      return float.NaN;
    }

    MGPosition nodePos = node.CalcPosition();
    int numPolicy = node.NumPolicyMoves;
    int numExpanded = node.NumEdgesExpanded;
    for (int i = 0; i < numPolicy; i++)
    {
      if (i < numExpanded)
      {
        GEdge edge = node.ChildEdgeAtIndex(i);
        if (edge.MoveMGFromPos(in nodePos) == move)
        {
          return edge.P;
        }
      }
      else
      {
        Graphs.GEdgeHeaders.GEdgeHeaderStruct header = node.EdgeHeadersSpan[i];
        if (Chess.MoveGen.Converters.ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(header.Move, in nodePos) == move)
        {
          return header.P;
        }
      }
    }

    return float.NaN;
  }


  /// <summary>
  /// Raw encoded move of the search root's most-visited expanded edge
  /// (cheap best-move snapshot for the root-flip impact counter).
  /// </summary>
  #region Root adaptive directed visits (LOCUS rootadapt)

  // One campaign per search: once the budget fraction has elapsed and the latest root probe
  // prefers a move other than the current top-N move, force visits into that move one chunk per
  // pump until it is best-by-Q at the root (flip) or the cap is reached (roll back).
  bool rootAdaptDone;
  bool rootAdaptActive;
  bool rootAdaptFlipped;
  int rootAdaptSlot = -1;
  ushort rootAdaptMoveRaw = ushort.MaxValue;
  int rootAdaptVisitsApplied;
  int rootAdaptCap;


  /// <summary>Fraction of this search's budget consumed so far (1 when unknown).</summary>
  double SearchProgressFraction()
  {
    SearchLimit limit = Manager.SearchLimit;
    if (limit == null || limit.Value <= 0)
    {
      return 1;
    }
    if (limit.Type == SearchLimitType.NodesPerTree)
    {
      double remaining = limit.Value - Manager.RootNWhenSearchStarted;
      return remaining <= 0 ? 1 : Manager.NumNodesVisitedThisSearch / remaining;
    }
    if (limit.IsNodesLimit)
    {
      return Manager.NumNodesVisitedThisSearch / (double)limit.Value;
    }
    return (DateTime.Now - Manager.StartTimeThisSearch).TotalSeconds / limit.Value;
  }


  void PumpRootAdapt(MCGSIterator iterator)
  {
    MCGSEngine engine = iterator.Engine;
    GNode root = engine.SearchRootNode;

    if (!rootAdaptActive)
    {
      if (!LastRootVerdict.HasValue || SearchProgressFraction() < Params.RootAdaptTriggerFraction)
      {
        return;
      }

      MGPosition rootPos = engine.SearchRootPosMG;
      int slot = ProbeGraftScheduler.ResolveMoveToChildIndex(root, in rootPos, LastRootVerdict.ProbeBestMove);
      if (slot < 0)
      {
        rootAdaptDone = true;
        return;
      }
      ushort probeRaw = (slot < root.NumEdgesExpanded ? root.ChildEdgeAtIndex(slot).Move
                                                        : root.EdgeHeadersSpan[slot].Move).RawValue;
      if (probeRaw == TopNRootMoveRaw(engine))
      {
        rootAdaptDone = true;   // the search already prefers the probe's move
        return;
      }

      SearchLimit limit = Manager.SearchLimit;
      int cap = Params.RootAdaptMaxVisits;
      if (limit != null && limit.IsNodesLimit && limit.Value > 0)
      {
        cap = Math.Min(cap, Math.Max(Params.RootAdaptVisitsPerPump, (int)(Params.RootAdaptMaxVisitsFraction * limit.Value)));
      }

      rootAdaptActive = true;
      rootAdaptSlot = slot;
      rootAdaptMoveRaw = probeRaw;
      rootAdaptVisitsApplied = 0;
      rootAdaptCap = cap;
      Stats.NumRootAdaptTriggers++;
    }

    int chunk = Math.Min(Params.RootAdaptVisitsPerPump, rootAdaptCap - rootAdaptVisitsApplied);
    DirectedVisits.Outcome step = chunk > 0
      ? DirectedVisits.ForceChildVisits(iterator, root, rootAdaptSlot, chunk)
      : new DirectedVisits.Outcome(0, 0, 0, "CapReached");
    rootAdaptVisitsApplied += step.VisitsApplied;
    Stats.NumRootAdaptVisits += step.VisitsApplied;

    bool decided = false;
    if (rootAdaptSlot < root.NumEdgesExpanded)
    {
      GEdge edge = root.ChildEdgeAtIndex(rootAdaptSlot);
      if (edge.Move.RawValue != rootAdaptMoveRaw)
      {
        FinishRootAdapt(engine, abandoned: true);   // slot reordered: abandon (rollback)
        return;
      }
      if (edge.Type != GEdgeStruct.EdgeType.Uninitialized
          && -edge.Q >= DirectedVisits.BestSiblingValue(root, rootAdaptSlot) + Params.RootAdaptMarginQ)
      {
        rootAdaptFlipped = true;
        decided = true;
      }
    }

    if (decided)
    {
      Stats.NumRootAdaptFlips++;
      rootAdaptActive = false;
      rootAdaptDone = true;
      if (Params.VerboseEventLogging)
      {
        Console.WriteLine($"[ROOTADAPT] flip after {rootAdaptVisitsApplied} visits (cap {rootAdaptCap})");
      }
      return;
    }

    if (step.Abort != null || step.VisitsApplied == 0 || rootAdaptVisitsApplied >= rootAdaptCap)
    {
      FinishRootAdapt(engine, abandoned: true);
    }
  }


  /// <summary>Ends an unflipped campaign: rolls its visits back (if configured) and marks it done.</summary>
  void FinishRootAdapt(MCGSEngine engine, bool abandoned)
  {
    if (rootAdaptActive && abandoned && !rootAdaptFlipped && Params.RootAdaptRollback && rootAdaptVisitsApplied > 0)
    {
      GNode root = engine.SearchRootNode;
      if (rootAdaptSlot >= 0 && rootAdaptSlot < root.NumEdgesExpanded
          && root.ChildEdgeAtIndex(rootAdaptSlot).Move.RawValue == rootAdaptMoveRaw)
      {
        DirectedVisits.ForgetEdgeVisits(root, rootAdaptSlot, rootAdaptVisitsApplied);
      }
      Stats.NumRootAdaptRollbacks++;
      if (Params.VerboseEventLogging)
      {
        Console.WriteLine($"[ROOTADAPT] no flip after {rootAdaptVisitsApplied} visits (cap {rootAdaptCap}); rolled back");
      }
    }
    rootAdaptActive = false;
    rootAdaptDone = true;
  }

  #endregion


  static ushort TopNRootMoveRaw(MCGSEngine engine)
  {
    int bestN = -1;
    ushort raw = ushort.MaxValue;
    foreach (GEdge edge in engine.SearchRootNode.ChildEdgesExpanded)
    {
      if (edge.N > bestN)
      {
        bestN = edge.N;
        raw = edge.Move.RawValue;
      }
    }
    return raw;
  }


  /// <summary>
  /// Milliseconds elapsed since a Stopwatch.GetTimestamp reading.
  /// </summary>
  /// <param name="startTicks"></param>
  static double MsSince(long startTicks)
    => (Stopwatch.GetTimestamp() - startTicks) * 1000.0 / Stopwatch.Frequency;


  /// <summary>
  /// Records one event line (verbose console echo plus the bounded dump-block log).
  /// </summary>
  /// <param name="message"></param>
  internal void LogEvent(string message)
  {
    if (!Params.VerboseEventLogging)
    {
      return;
    }

    if (eventLog.Count < MAX_EVENT_LOG_LINES)
    {
      eventLog.Add(message);
    }
    ConsoleUtils.WriteLineColored(ConsoleColor.DarkCyan, "[ProbeGraft] " + message);
  }


  void DisableAfterError(Exception exc)
  {
    disabledByError = true;
    Stats.DisabledByError = true;
    Stats.DisabledByErrorMessage = exc.Message;
    ConsoleUtils.WriteLineColored(ConsoleColor.Red,
      "[ProbeGraft] disabled for remainder of search after exception: " + exc.Message);
  }


  void ShutdownSession()
  {
    try
    {
      cts.Cancel();
      session?.EndSession();
    }
    catch (Exception exc)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Red, "[ProbeGraft] EndSession failed: " + exc.Message);
    }
  }

  #endregion
}
