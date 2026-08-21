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

#endregion

namespace Ceres.MCGS.Search.Params;

/// <summary>
/// Parameters for the opt-in "refutation grafting" hybrid feature (PICKET M1).
///
/// Important interior nodes of the live search graph are asynchronously verified by an
/// external exact-search probe source (see ProbeSourceRegistry and the
/// Ceres.Chess.Probing.IProbeSource contract). When a probe refutes the move
/// currently dominating a node's visit distribution (or surfaces a strong move the policy
/// prior ignores), the probe's witness line is grafted into the graph as forced, committed,
/// NN-evaluated visits executed through the stock select/evaluate/backup pipeline, plus an
/// optional bounded prior nudge. No probe-derived value is ever written into the graph: the NN
/// evaluations of the grafted line are the sole evidence and ordinary backup draws the
/// conclusion. Probe scores are consulted only as trigger thresholds.
///
/// Disabled by default (a true no-op). Shadow mode runs targeting, probes and trigger
/// decisions and logs end-of-search verdicts without ever mutating the graph.
/// </summary>
[Serializable]
public record ParamsProbeGraft
{
  /// <summary>
  /// Constructor.
  /// </summary>
  public ParamsProbeGraft()
  {
  }

  /// <summary>
  /// Operating mode for the feature.
  /// </summary>
  public enum ModeType
  {
    /// <summary>Disabled: no targeting, no probes, no grafts (default; a true no-op).</summary>
    Disabled,

    /// <summary>
    /// Shadow: targeting and probes run and all decisions are computed and logged
    /// (including end-of-search verdicts), but no graft visits and no prior nudges
    /// are applied - the graph is never mutated by the feature.
    ///
    /// Verdict interpretation: a shadow event CONFIRMED by search is one plain search
    /// resolved anyway (a graft would only have accelerated it); UNRESOLVED events are
    /// the feature's potential value - they are where Active mode differs from baseline.
    /// </summary>
    Shadow,

    /// <summary>Active: grafts and prior nudges are applied.</summary>
    Active
  };

  /// <summary>Operating mode (Disabled / Shadow / Active).</summary>
  public ModeType Mode = ModeType.Disabled;

  /// <summary>
  /// ID of the probe source registered with ProbeSourceRegistry. Null/empty with
  /// Mode != Disabled is a validation error (there is nothing to probe with).
  /// </summary>
  public string ProbeSourceID = null;

  #region Targeting (region walk; see ProbeGraftTargeter)

  /// <summary>Seconds between target-selection sweeps (also min spacing of probe issuance).</summary>
  public float TargetSweepIntervalSeconds = 0.5f;

  /// <summary>Max plies below the search root considered by the target sweep.</summary>
  public int TargetMaxDepth = 12;

  /// <summary>Node visit-count floor as a fraction of root N (mirrors CutFraction in RootMinimaxBlend).</summary>
  public float TargetCutFraction = 0.002f;

  /// <summary>Absolute node visit-count floor for targeting.</summary>
  public int TargetNCutAbs = 500;

  /// <summary>Max nodes selected per sweep (top by priority score).</summary>
  public int TargetMaxNodesPerSweep = 64;

  /// <summary>Max in-flight (issued, uncompleted) probe requests.</summary>
  public int MaxInFlightProbes = 256;

  /// <summary>
  /// Per-node minimum re-probe spacing, as a multiple of growth in the node's N
  /// (a node is re-probed only after its N has grown by this factor since its last probe).
  /// </summary>
  public float ReprobeNGrowthFactor = 2.0f;

  /// <summary>Node budget passed to the probe source per request.</summary>
  public int ProbeNodeBudget = 200_000;

  /// <summary>
  /// If the search root itself is eligible as a probe target. The root's probe asks the one
  /// question that maps directly onto the move actually played ("is the move we are about to
  /// play refuted?"), so it carries by far the most decision leverage; every other target can
  /// only matter to the extent its value propagates up to a root child. Off by default because
  /// acting on a root refutation is closer to a root veto than to ordinary grafting.
  /// </summary>
  public bool TargetIncludeSearchRoot = false;

  /// <summary>
  /// Max depth below the search root at which the shallow (faster) re-probe cadence applies.
  /// 0 restricts it to the search root itself, so with TargetIncludeSearchRoot false the shallow
  /// cadence is inert and re-probe spacing is uniformly ReprobeNGrowthFactor.
  /// </summary>
  public int ReprobeShallowMaxDepth = 0;

  /// <summary>
  /// Re-probe spacing (as a multiple of growth in the node's N) for nodes at or above
  /// ReprobeShallowMaxDepth. Because ReprobeNGrowthFactor is geometric, a node is probed only
  /// O(log N) times across a search, which saturates coverage of the few high-leverage nodes near
  /// the root long before probe capacity is exhausted. A smaller factor here re-probes those nodes
  /// as the search evolves. Ignored when it exceeds ReprobeNGrowthFactor.
  /// </summary>
  public float ReprobeNGrowthFactorShallow = 1.10f;

  /// <summary>
  /// How candidate targets are ranked within a sweep.
  /// </summary>
  public enum TargetPriorityModeType
  {
    /// <summary>By node visit count (highest N first), zero when the dominant-visit gate fails.</summary>
    VisitCount,

    /// <summary>
    /// By estimated proximity to changing the root's move choice: for a candidate X under root
    /// child C, the value change required at X to reorder the root is approximately
    /// (gap between C and the best alternative root child) * N(C) / N(X). Candidates are ranked by
    /// ascending required change, which folds leverage and root urgency into one score and demotes
    /// deep, low-influence nodes without a hard depth cap.
    /// </summary>
    RootFlipDistance
  }

  /// <summary>Ranking rule used to order candidates within a target sweep.</summary>
  public TargetPriorityModeType TargetPriorityMode = TargetPriorityModeType.VisitCount;

  /// <summary>
  /// RootFlipDistance mode only: candidates whose estimated required value change exceeds this
  /// (in Q units, so the full range is 2.0) are not probed at all, on the grounds that no plausible
  /// refutation there could change the move played. Values >= 2 disable the gate.
  /// </summary>
  public float TargetMaxRequiredDelta = 2.0f;

  #endregion

  #region Graft triggering (probe result interpretation)

  /// <summary>
  /// Minimum centipawn margin by which the probe must refute the dominant move at the node
  /// (fail-low of the dominant move / fail-high of an alternative, per the probe's own scores;
  /// used ONLY as a trigger threshold, never converted into a graph value).
  /// </summary>
  public int TriggerMarginCp = 200;

  /// <summary>Minimum probe search depth (iterations) for a result to be trusted as a trigger.</summary>
  public int TriggerMinProbeDepth = 10;

  /// <summary>
  /// Fraction of the node's visits the dominant child must hold for "refuted dominant move"
  /// triggering to apply (below this the node has no meaningful dominant line to refute).
  /// </summary>
  public float DominantChildMinVisitFraction = 0.45f;

  #endregion

  #region Graft application

  /// <summary>Maximum plies of the witness PV grafted (spine length cap).</summary>
  public int GraftMaxPlies = 16;

  /// <summary>Committed visits sent through each grafted ply (the ply's forced ProbeSpec NumVisits).</summary>
  public int GraftVisitsPerPly = 1;

  /// <summary>Maximum concurrently advancing grafts.</summary>
  public int MaxActiveGrafts = 32;

  /// <summary>
  /// Hard cap on total NN evaluations consumed by grafting per search, as a fraction of
  /// evaluations performed so far (grafting pauses when the cap is reached). Guards GPU budget.
  /// </summary>
  public float MaxGraftEvalFraction = 0.03f;

  /// <summary>
  /// If nonzero, the prior of the probe's refutation move at each grafted ply is raised to at
  /// least this value (renormalizing the node's other priors down proportionally), capped by
  /// PriorNudgeMaxTotalMass per node over the search. 0 disables nudging entirely.
  /// Also serves as the low-prior threshold of the secondary "discovery" trigger (see
  /// ProbeGraftCoordinator): a probe best move whose prior is below this value is one the
  /// policy effectively ignores.
  /// </summary>
  public float PriorNudgeFloor = 0.03f;

  /// <summary>Cap on cumulative prior mass moved by nudges at any single node.</summary>
  public float PriorNudgeMaxTotalMass = 0.15f;

  /// <summary>
  /// If true, edges along a completed graft path are marked stale so that a configured
  /// PostBackupQMode == StaleDrain pass propagates the graft's evidence upward promptly.
  /// No effect (and validated as an error) unless PostBackupQMode == StaleDrain.
  /// </summary>
  public bool MarkGraftPathStale = false;

  #endregion

  #region Diagnostics

  /// <summary>If a one-line per-search summary of graft statistics is printed at end of search.</summary>
  public bool EnableStatsSummary = true;

  /// <summary>If per-event (probe/graft) detail lines are logged (verbose; Shadow-mode tuning aid).</summary>
  public bool VerboseEventLogging = false;

  #endregion

  /// <summary>
  /// Validates settings for self-consistency (including cross-checks against the parent
  /// ParamsSearch). Called from ParamsSearch.Validate(). All checks apply only when the
  /// feature is not Disabled.
  /// </summary>
  /// <param name="parent"></param>
  public void Validate(ParamsSearch parent)
  {
    if (Mode == ModeType.Disabled)
    {
      return;
    }

    if (string.IsNullOrEmpty(ProbeSourceID))
    {
      throw new Exception("ParamsProbeGraft.ProbeSourceID must be set when Mode is not Disabled.");
    }

    if (parent.Execution.DualOverlappedIterators)
    {
      // The RunProbeSpecs contract requires single-iterator harnesses (quiescent graph at pump time).
      throw new Exception("ParamsProbeGraft cannot be used with Execution.DualOverlappedIterators.");
    }

    if (MarkGraftPathStale && parent.PostBackupQMode != ParamsSearch.PostBackupQModeType.StaleDrain)
    {
      throw new Exception("ParamsProbeGraft.MarkGraftPathStale requires PostBackupQMode == StaleDrain.");
    }

    if (GraftMaxPlies < 1)
    {
      throw new Exception("ParamsProbeGraft.GraftMaxPlies must be at least 1.");
    }

    if (GraftVisitsPerPly < 1)
    {
      throw new Exception("ParamsProbeGraft.GraftVisitsPerPly must be at least 1.");
    }

    if (TriggerMarginCp < 0)
    {
      throw new Exception("ParamsProbeGraft.TriggerMarginCp must be non-negative.");
    }

    if (PriorNudgeFloor < 0 || PriorNudgeFloor > 0.2f)
    {
      throw new Exception("ParamsProbeGraft.PriorNudgeFloor must be in [0, 0.2].");
    }

    if (PriorNudgeFloor > 0 && (PriorNudgeMaxTotalMass <= 0 || PriorNudgeMaxTotalMass > 0.5f))
    {
      throw new Exception("ParamsProbeGraft.PriorNudgeMaxTotalMass must be in (0, 0.5] when nudging enabled.");
    }

    if (MaxGraftEvalFraction <= 0 || MaxGraftEvalFraction > 0.25f)
    {
      throw new Exception("ParamsProbeGraft.MaxGraftEvalFraction must be in (0, 0.25].");
    }

    if (TargetCutFraction <= 0 || TargetCutFraction > 0.5f)
    {
      throw new Exception("ParamsProbeGraft.TargetCutFraction must be in (0, 0.5].");
    }

    if (TargetNCutAbs < 1)
    {
      throw new Exception("ParamsProbeGraft.TargetNCutAbs must be at least 1.");
    }

    if (TargetSweepIntervalSeconds <= 0)
    {
      throw new Exception("ParamsProbeGraft.TargetSweepIntervalSeconds must be positive.");
    }

    if (TargetMaxDepth < 1)
    {
      throw new Exception("ParamsProbeGraft.TargetMaxDepth must be at least 1.");
    }

    if (TargetMaxNodesPerSweep < 1)
    {
      throw new Exception("ParamsProbeGraft.TargetMaxNodesPerSweep must be at least 1.");
    }

    if (MaxInFlightProbes < 1)
    {
      throw new Exception("ParamsProbeGraft.MaxInFlightProbes must be at least 1.");
    }

    if (ReprobeNGrowthFactor < 1)
    {
      throw new Exception("ParamsProbeGraft.ReprobeNGrowthFactor must be at least 1.");
    }

    if (ReprobeNGrowthFactorShallow < 1)
    {
      throw new Exception("ParamsProbeGraft.ReprobeNGrowthFactorShallow must be at least 1.");
    }

    if (ReprobeShallowMaxDepth < 0)
    {
      throw new Exception("ParamsProbeGraft.ReprobeShallowMaxDepth cannot be negative.");
    }

    if (TargetMaxRequiredDelta <= 0)
    {
      throw new Exception("ParamsProbeGraft.TargetMaxRequiredDelta must be positive.");
    }

    if (ProbeNodeBudget < 1)
    {
      throw new Exception("ParamsProbeGraft.ProbeNodeBudget must be at least 1.");
    }

    if (MaxActiveGrafts < 1)
    {
      throw new Exception("ParamsProbeGraft.MaxActiveGrafts must be at least 1.");
    }
  }
}
