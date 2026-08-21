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
using System.IO;
using System.Text;

using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Search.ProbeGraft;

/// <summary>
/// Per-search statistics of the PICKET M1 refutation-grafting feature (usage counters,
/// timing, probe-side latency, and impact estimates), owned by the ProbeGraftCoordinator and
/// exposed as MCGSManager.ProbeGraftStats for external diagnostics.
///
/// Plain class with plain fields: all mutation occurs on the pump thread (the search
/// thread, at the post-batch quiescent point) - the probe completion callbacks never touch
/// this object (they only enqueue results into the coordinator's inbox).
///
/// Impact figures are correlational, not causal (Shadow mode plus A/B testing is the
/// causal instrument); they are tuning signals only.
/// </summary>
public sealed class ProbeGraftStats
{
  #region Usage counters

  /// <summary>Number of target-selection sweeps performed.</summary>
  public long NumTargetSweeps;

  /// <summary>Number of target nodes selected across all sweeps (before submission capping).</summary>
  public long NumTargetsSelected;

  /// <summary>Number of probe requests accepted by the source.</summary>
  public long NumProbesSubmitted;

  /// <summary>Number of probe requests the source declined to accept (queue full/unavailable).</summary>
  public long NumProbesRejectedBySource;

  /// <summary>Number of results received with Completed == true.</summary>
  public long NumProbesCompleted;

  /// <summary>Number of results received with Completed == false (cancelled/errored/degenerate).</summary>
  public long NumProbesIncomplete;

  /// <summary>Completed results discarded because the target snapshot no longer matched the graph.</summary>
  public long NumResultsStale;

  /// <summary>Results still in the inbox when the search ended (drained and discarded).</summary>
  public long NumResultsDiscardedAtEnd;

  /// <summary>Completed results where the probe agreed with the node (no trigger; the expected common case).</summary>
  public long NumProbesAgreed;

  /// <summary>Refutation triggers (probe refuted the visit-dominant move by at least the margin).</summary>
  public long NumTriggersRefutation;

  /// <summary>Discovery triggers (probe best move has prior below the nudge floor).</summary>
  public long NumTriggersDiscovery;

  /// <summary>Grafts promoted to actively advancing (Active mode only).</summary>
  public long NumGraftsStarted;

  /// <summary>Grafts that applied their full move list (or reached GraftMaxPlies).</summary>
  public long NumGraftsCompleted;

  /// <summary>Grafts that ended by running into a terminal (mate/draw) - a successful completion.</summary>
  public long NumGraftsEndedTerminal;

  /// <summary>Grafts aborted because a witness move was absent from the node's policy move list.</summary>
  public long NumGraftsAbortedMoveNotFound;

  /// <summary>Grafts aborted because their forced path evaporated (dropped/aborted visits).</summary>
  public long NumGraftsAbortedPathDropped;

  /// <summary>Pumps on which graft advancement was paused by the MaxGraftEvalFraction budget.</summary>
  public long NumGraftsPausedBudget;

  /// <summary>Total grafted plies applied (committed forced visits through a witness move).</summary>
  public long NumGraftPliesApplied;

  /// <summary>Total NN evaluations consumed by graft advancement (budget numerator).</summary>
  public long NumGraftNNEvals;

  /// <summary>Prior nudges applied (one per nudged ply).</summary>
  public long NumPriorNudgesApplied;

  /// <summary>Prior nudges skipped or clamped by the per-node cumulative mass cap.</summary>
  public long NumPriorNudgesCappedByMass;

  /// <summary>If the feature disabled itself for the remainder of the search after an exception.</summary>
  public bool DisabledByError;

  /// <summary>Message of the first exception that disabled the feature (null if none).</summary>
  public string DisabledByErrorMessage;

  #endregion

  #region Timing (milliseconds, accumulated on the pump thread via Stopwatch.GetTimestamp deltas)

  /// <summary>Total time spent inside non-idle pumps (the search-thread cost of the feature).</summary>
  public double TimePumpTotalMs;

  /// <summary>Portion of pump time spent draining/interpreting probe results.</summary>
  public double TimeDrainMs;

  /// <summary>Portion of pump time spent advancing grafts (including the RunProbeSpecs calls).</summary>
  public double TimeGraftAdvanceMs;

  /// <summary>Portion of pump time spent in target sweeps and probe submission.</summary>
  public double TimeTargetSweepMs;

  /// <summary>Sum of probe wall latencies (from result metadata; for averages).</summary>
  public double ProbeLatencySumMs;

  /// <summary>Maximum probe wall latency observed.</summary>
  public double ProbeLatencyMaxMs;

  /// <summary>Sum of prober node counts over completed probes.</summary>
  public long ProbeNodesSum;

  /// <summary>Sum of prober iteration depths over completed probes.</summary>
  public long ProbeDepthSum;

  #endregion

  #region Impact (correlational)

  /// <summary>Sum over completed grafts of |anchor Q at end of search - anchor Q before first grafted ply|.</summary>
  public double SumAbsAnchorQDeltaAtEnd;

  /// <summary>Maximum |anchor Q delta| over completed grafts.</summary>
  public double MaxAbsAnchorQDeltaAtEnd;

  /// <summary>Completed grafts whose |anchor Q delta| exceeded 0.03 (BottomUpQRecalculator.LARGE_DELTA_THRESHOLD).</summary>
  public long NumAnchorsQMovedOverThreshold;

  /// <summary>Times the root's top-N move changed across a pump that applied graft plies.</summary>
  public long NumRootBestMoveChangesAfterGraft;

  #endregion

  #region Shadow verdicts

  /// <summary>Shadow mode: number of would-have-grafted events recorded.</summary>
  public long ShadowNumEvents;

  /// <summary>Shadow mode: events plain search resolved on its own by end of search.</summary>
  public long ShadowNumConfirmedBySearch;

  /// <summary>Shadow mode: events left unresolved by plain search (the feature's potential value).</summary>
  public long ShadowNumUnresolved;

  #endregion



  #region Target depth distribution (diagnostic)

  /// <summary>Number of depth bins tracked below the search root (deeper targets fold into the last bin).</summary>
  public const int MAX_DEPTH_BINS = 16;

  /// <summary>Probes submitted, binned by the target node's depth below the search root.</summary>
  public long[] ProbesSubmittedAtDepth = new long[MAX_DEPTH_BINS];

  /// <summary>Triggers (refutation plus discovery), binned by the target node's depth below the search root.</summary>
  public long[] TriggersAtDepth = new long[MAX_DEPTH_BINS];

  /// <summary>Number of distinct graph nodes ever probed during the search (probes per node = submitted / this).</summary>
  public long NumDistinctNodesProbed;

  /// <summary>
  /// Renders the depth histograms as "dN:probes/triggers" over the occupied bins.
  /// </summary>
  /// <returns></returns>
  public string DepthHistogramLine()
  {
    StringBuilder builder = new();
    builder.Append("[ProbeGraft-depth] distinctNodes=" + NumDistinctNodesProbed + " probes/triggers by depth:");
    for (int depth = 0; depth < MAX_DEPTH_BINS; depth++)
    {
      if (ProbesSubmittedAtDepth[depth] > 0 || TriggersAtDepth[depth] > 0)
      {
        builder.Append($" d{depth}:{ProbesSubmittedAtDepth[depth]}/{TriggersAtDepth[depth]}");
      }
    }
    return builder.ToString();
  }


  /// <summary>
  /// Records one submitted probe against its target's depth below the search root.
  /// </summary>
  /// <param name="depth"></param>
  public void RecordProbeDepth(int depth) => ProbesSubmittedAtDepth[Math.Clamp(depth, 0, MAX_DEPTH_BINS - 1)]++;


  /// <summary>
  /// Records one trigger against its target's depth below the search root.
  /// </summary>
  /// <param name="depth"></param>
  public void RecordTriggerDepth(int depth) => TriggersAtDepth[Math.Clamp(depth, 0, MAX_DEPTH_BINS - 1)]++;

  #endregion


  #region Aggregation across searches

  /// <summary>
  /// Adds all counters of another instance into this one (sums, except the max-valued
  /// fields which are maxed). Used to aggregate per-search statistics over a set of
  /// searches (a measurement campaign, a game, a match).
  /// </summary>
  /// <param name="other"></param>
  public void AccumulateFrom(ProbeGraftStats other)
  {
    if (other == null)
    {
      return;
    }

    NumTargetSweeps += other.NumTargetSweeps;
    NumTargetsSelected += other.NumTargetsSelected;
    NumProbesSubmitted += other.NumProbesSubmitted;
    NumProbesRejectedBySource += other.NumProbesRejectedBySource;
    NumProbesCompleted += other.NumProbesCompleted;
    NumProbesIncomplete += other.NumProbesIncomplete;
    NumResultsStale += other.NumResultsStale;
    NumResultsDiscardedAtEnd += other.NumResultsDiscardedAtEnd;
    NumProbesAgreed += other.NumProbesAgreed;
    NumTriggersRefutation += other.NumTriggersRefutation;
    NumTriggersDiscovery += other.NumTriggersDiscovery;
    NumGraftsStarted += other.NumGraftsStarted;
    NumGraftsCompleted += other.NumGraftsCompleted;
    NumGraftsEndedTerminal += other.NumGraftsEndedTerminal;
    NumGraftsAbortedMoveNotFound += other.NumGraftsAbortedMoveNotFound;
    NumGraftsAbortedPathDropped += other.NumGraftsAbortedPathDropped;
    NumGraftsPausedBudget += other.NumGraftsPausedBudget;
    NumGraftPliesApplied += other.NumGraftPliesApplied;
    NumGraftNNEvals += other.NumGraftNNEvals;
    NumPriorNudgesApplied += other.NumPriorNudgesApplied;
    NumPriorNudgesCappedByMass += other.NumPriorNudgesCappedByMass;

    TimePumpTotalMs += other.TimePumpTotalMs;
    TimeDrainMs += other.TimeDrainMs;
    TimeGraftAdvanceMs += other.TimeGraftAdvanceMs;
    TimeTargetSweepMs += other.TimeTargetSweepMs;
    ProbeLatencySumMs += other.ProbeLatencySumMs;
    ProbeLatencyMaxMs = Math.Max(ProbeLatencyMaxMs, other.ProbeLatencyMaxMs);
    ProbeNodesSum += other.ProbeNodesSum;
    ProbeDepthSum += other.ProbeDepthSum;

    SumAbsAnchorQDeltaAtEnd += other.SumAbsAnchorQDeltaAtEnd;
    MaxAbsAnchorQDeltaAtEnd = Math.Max(MaxAbsAnchorQDeltaAtEnd, other.MaxAbsAnchorQDeltaAtEnd);
    NumAnchorsQMovedOverThreshold += other.NumAnchorsQMovedOverThreshold;
    NumRootBestMoveChangesAfterGraft += other.NumRootBestMoveChangesAfterGraft;

    NumDistinctNodesProbed += other.NumDistinctNodesProbed;
    for (int depth = 0; depth < MAX_DEPTH_BINS; depth++)
    {
      ProbesSubmittedAtDepth[depth] += other.ProbesSubmittedAtDepth[depth];
      TriggersAtDepth[depth] += other.TriggersAtDepth[depth];
    }

    ShadowNumEvents += other.ShadowNumEvents;
    ShadowNumConfirmedBySearch += other.ShadowNumConfirmedBySearch;
    ShadowNumUnresolved += other.ShadowNumUnresolved;

    if (other.DisabledByError && !DisabledByError)
    {
      DisabledByError = true;
      DisabledByErrorMessage = other.DisabledByErrorMessage;
    }
  }

  #endregion

  #region Process-wide accumulation (multi-search harnesses)

  /// <summary>
  /// If per-search statistics should be accumulated into a process-wide aggregate as each
  /// search ends. Intended for harnesses that run many searches (games, matches, measurement
  /// campaigns, tournament managers) and cannot conveniently collect per-search objects.
  /// Off by default; the aggregate is inert unless explicitly enabled.
  /// </summary>
  public static bool EnableGlobalAccumulation = false;

  static readonly object globalLockObj = new();
  static readonly ProbeGraftStats globalStats = new();
  static int globalNumSearches;
  static long globalNumEvals;
  static ParamsProbeGraft.ModeType globalMode = ParamsProbeGraft.ModeType.Disabled;

  /// <summary>
  /// Clears the process-wide aggregate (call before each measurement arm).
  /// </summary>
  public static void ResetGlobal()
  {
    lock (globalLockObj)
    {
      globalStats.ClearAll();
      globalNumSearches = 0;
      globalNumEvals = 0;
      globalMode = ParamsProbeGraft.ModeType.Disabled;
    }
  }


  /// <summary>
  /// Adds the statistics of one completed search into the process-wide aggregate.
  /// </summary>
  /// <param name="stats"></param>
  /// <param name="mode"></param>
  /// <param name="numEvalsThisSearch"></param>
  public static void AccumulateGlobal(ProbeGraftStats stats, ParamsProbeGraft.ModeType mode, int numEvalsThisSearch)
  {
    lock (globalLockObj)
    {
      globalStats.AccumulateFrom(stats);
      globalNumSearches++;
      globalNumEvals += numEvalsThisSearch;
      globalMode = mode;
    }
  }


  /// <summary>
  /// Returns a snapshot copy of the process-wide aggregate, the number of searches
  /// contributing to it, and the total NN evaluations across those searches.
  /// </summary>
  /// <returns></returns>
  public static (ProbeGraftStats Stats, int NumSearches, long NumEvals) GlobalSnapshot()
  {
    lock (globalLockObj)
    {
      ProbeGraftStats copy = new();
      copy.AccumulateFrom(globalStats);
      return (copy, globalNumSearches, globalNumEvals);
    }
  }


  /// <summary>
  /// Renders a one-line summary of the process-wide aggregate (same layout as the
  /// per-search summary, prefixed with the number of contributing searches).
  /// </summary>
  /// <returns></returns>
  public static string GlobalSummaryLine()
  {
    (ProbeGraftStats stats, int numSearches, long numEvals) = GlobalSnapshot();
    int evalsForLine = (int)Math.Min(numEvals, int.MaxValue);
    return $"[ProbeGraft-TOTAL over {numSearches} searches] " + stats.SummaryLine(globalMode, evalsForLine);
  }


  /// <summary>
  /// Resets every field to its default value.
  /// </summary>
  void ClearAll()
  {
    NumTargetSweeps = 0;
    NumTargetsSelected = 0;
    NumProbesSubmitted = 0;
    NumProbesRejectedBySource = 0;
    NumProbesCompleted = 0;
    NumProbesIncomplete = 0;
    NumResultsStale = 0;
    NumResultsDiscardedAtEnd = 0;
    NumProbesAgreed = 0;
    NumTriggersRefutation = 0;
    NumTriggersDiscovery = 0;
    NumGraftsStarted = 0;
    NumGraftsCompleted = 0;
    NumGraftsEndedTerminal = 0;
    NumGraftsAbortedMoveNotFound = 0;
    NumGraftsAbortedPathDropped = 0;
    NumGraftsPausedBudget = 0;
    NumGraftPliesApplied = 0;
    NumGraftNNEvals = 0;
    NumPriorNudgesApplied = 0;
    NumPriorNudgesCappedByMass = 0;
    DisabledByError = false;
    DisabledByErrorMessage = null;
    TimePumpTotalMs = 0;
    TimeDrainMs = 0;
    TimeGraftAdvanceMs = 0;
    TimeTargetSweepMs = 0;
    ProbeLatencySumMs = 0;
    ProbeLatencyMaxMs = 0;
    ProbeNodesSum = 0;
    ProbeDepthSum = 0;
    SumAbsAnchorQDeltaAtEnd = 0;
    MaxAbsAnchorQDeltaAtEnd = 0;
    NumAnchorsQMovedOverThreshold = 0;
    NumRootBestMoveChangesAfterGraft = 0;
    ShadowNumEvents = 0;
    ShadowNumConfirmedBySearch = 0;
    ShadowNumUnresolved = 0;
    NumDistinctNodesProbed = 0;
    Array.Clear(ProbesSubmittedAtDepth);
    Array.Clear(TriggersAtDepth);
  }

  #endregion


  /// <summary>
  /// Renders the one-line end-of-search summary (see ProbeGraftCoordinator.EndSearch).
  /// </summary>
  /// <param name="mode"></param>
  /// <param name="numEvalsThisSearch"></param>
  /// <returns></returns>
  public string SummaryLine(ParamsProbeGraft.ModeType mode, int numEvalsThisSearch)
  {
    double graftEvalPct = numEvalsThisSearch > 0 ? 100.0 * NumGraftNNEvals / numEvalsThisSearch : 0;
    double probeLatAvg = NumProbesCompleted > 0 ? ProbeLatencySumMs / NumProbesCompleted : 0;

    string line = $"[ProbeGraft] mode={mode} probes: sub={NumProbesSubmitted} done={NumProbesCompleted} "
                + $"agree={NumProbesAgreed} stale={NumResultsStale} | triggers: ref={NumTriggersRefutation} disc={NumTriggersDiscovery} "
                + $"grafts: start={NumGraftsStarted} done={NumGraftsCompleted} term={NumGraftsEndedTerminal} "
                + $"abort={NumGraftsAbortedMoveNotFound + NumGraftsAbortedPathDropped} "
                + $"(mnf={NumGraftsAbortedMoveNotFound} pd={NumGraftsAbortedPathDropped} pause={NumGraftsPausedBudget}) "
                + $"plies={NumGraftPliesApplied} "
                + $"evals={NumGraftNNEvals} ({graftEvalPct:F1}% of {numEvalsThisSearch / 1000.0:F1}k) nudges={NumPriorNudgesApplied} "
                + $"| pump={TimePumpTotalMs:F1}ms (drain={TimeDrainMs:F1} graft={TimeGraftAdvanceMs:F1} sweep={TimeTargetSweepMs:F1}) "
                + $"probeLatAvg={probeLatAvg:F0}ms | impact: dQmax={MaxAbsAnchorQDeltaAtEnd:F3} "
                + $"movedGT.03={NumAnchorsQMovedOverThreshold} rootFlips={NumRootBestMoveChangesAfterGraft}";

    if (mode == ParamsProbeGraft.ModeType.Shadow)
    {
      line += $" | shadow: ev={ShadowNumEvents} conf={ShadowNumConfirmedBySearch} unres={ShadowNumUnresolved}";
    }

    if (DisabledByError)
    {
      line += $" | DISABLED-BY-ERROR: {DisabledByErrorMessage}";
    }

    return line;
  }


  /// <summary>
  /// Writes the full field dump (one "label : value" per line) for the diagnostics dump block.
  /// </summary>
  /// <param name="writer"></param>
  public void Dump(TextWriter writer)
  {
    writer.WriteLine("NumTargetSweeps               : " + NumTargetSweeps);
    writer.WriteLine("NumTargetsSelected            : " + NumTargetsSelected);
    writer.WriteLine("NumProbesSubmitted            : " + NumProbesSubmitted);
    writer.WriteLine("NumProbesRejectedBySource     : " + NumProbesRejectedBySource);
    writer.WriteLine("NumProbesCompleted            : " + NumProbesCompleted);
    writer.WriteLine("NumProbesIncomplete           : " + NumProbesIncomplete);
    writer.WriteLine("NumResultsStale               : " + NumResultsStale);
    writer.WriteLine("NumResultsDiscardedAtEnd      : " + NumResultsDiscardedAtEnd);
    writer.WriteLine("NumProbesAgreed               : " + NumProbesAgreed);
    writer.WriteLine("NumTriggersRefutation         : " + NumTriggersRefutation);
    writer.WriteLine("NumTriggersDiscovery          : " + NumTriggersDiscovery);
    writer.WriteLine("NumGraftsStarted              : " + NumGraftsStarted);
    writer.WriteLine("NumGraftsCompleted            : " + NumGraftsCompleted);
    writer.WriteLine("NumGraftsEndedTerminal        : " + NumGraftsEndedTerminal);
    writer.WriteLine("NumGraftsAbortedMoveNotFound  : " + NumGraftsAbortedMoveNotFound);
    writer.WriteLine("NumGraftsAbortedPathDropped   : " + NumGraftsAbortedPathDropped);
    writer.WriteLine("NumGraftsPausedBudget         : " + NumGraftsPausedBudget);
    writer.WriteLine("NumGraftPliesApplied          : " + NumGraftPliesApplied);
    writer.WriteLine("NumGraftNNEvals               : " + NumGraftNNEvals);
    writer.WriteLine("NumPriorNudgesApplied         : " + NumPriorNudgesApplied);
    writer.WriteLine("NumPriorNudgesCappedByMass    : " + NumPriorNudgesCappedByMass);
    writer.WriteLine("DisabledByError               : " + DisabledByError
                     + (DisabledByError ? " (" + DisabledByErrorMessage + ")" : ""));
    writer.WriteLine("TimePumpTotalMs               : " + TimePumpTotalMs.ToString("F2"));
    writer.WriteLine("TimeDrainMs                   : " + TimeDrainMs.ToString("F2"));
    writer.WriteLine("TimeGraftAdvanceMs            : " + TimeGraftAdvanceMs.ToString("F2"));
    writer.WriteLine("TimeTargetSweepMs             : " + TimeTargetSweepMs.ToString("F2"));
    writer.WriteLine("ProbeLatencySumMs             : " + ProbeLatencySumMs.ToString("F1"));
    writer.WriteLine("ProbeLatencyMaxMs             : " + ProbeLatencyMaxMs.ToString("F1"));
    writer.WriteLine("ProbeNodesSum                 : " + ProbeNodesSum);
    writer.WriteLine("ProbeDepthSum                 : " + ProbeDepthSum);
    writer.WriteLine("SumAbsAnchorQDeltaAtEnd       : " + SumAbsAnchorQDeltaAtEnd.ToString("F4"));
    writer.WriteLine("MaxAbsAnchorQDeltaAtEnd       : " + MaxAbsAnchorQDeltaAtEnd.ToString("F4"));
    writer.WriteLine("NumAnchorsQMovedOverThreshold : " + NumAnchorsQMovedOverThreshold);
    writer.WriteLine("NumRootBestMoveChangesAfterGraft : " + NumRootBestMoveChangesAfterGraft);
    writer.WriteLine("ShadowNumEvents               : " + ShadowNumEvents);
    writer.WriteLine("ShadowNumConfirmedBySearch    : " + ShadowNumConfirmedBySearch);
    writer.WriteLine("ShadowNumUnresolved           : " + ShadowNumUnresolved);
    writer.WriteLine(DepthHistogramLine());
  }
}
