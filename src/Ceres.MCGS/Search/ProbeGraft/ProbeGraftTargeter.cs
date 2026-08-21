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

using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Search.ProbeGraft;

/// <summary>
/// Immutable snapshot of one probe-worthy interior node taken during a target sweep:
/// its identity, visit statistics, dominant child, and the move path from the search root
/// discovered by the sweep's walk (an arbitrary DAG node has no canonical path, but the
/// walk's path is a valid one and is what the probe analyzes). Round-tripped to the probe
/// result as the request's opaque RequesterToken.
/// </summary>
public sealed class ProbeGraftTargetSnapshot
{
  /// <summary>Graph index of the target node.</summary>
  public int NodeIndex;

  /// <summary>Node visit count N at snapshot time.</summary>
  public int N;

  /// <summary>Node Q (own side-to-move perspective) at snapshot time.</summary>
  public double Q;

  /// <summary>Plies below the search root at which the walk found this node.</summary>
  public int Depth;

  /// <summary>Index (within the node's edges) of the child with the most visits.</summary>
  public int DominantChildIndex;

  /// <summary>Fraction of the node's N held by the dominant child.</summary>
  public float DominantVisitFraction;

  /// <summary>The dominant child's move (default when the node has no visited children).</summary>
  public MGMove DominantMove;

  /// <summary>If the dominant-fraction refutation gate held at snapshot time.</summary>
  public bool DominantFractionGateHeld;

  /// <summary>Move path from the search root to this node, as discovered by the walk.</summary>
  public MGMove[] MovesFromRoot;

  /// <summary>
  /// Sweep priority, highest first. In VisitCount mode this is N when the dominant-fraction gate
  /// holds and 0 otherwise; in RootFlipDistance mode it is the reciprocal of RequiredDelta.
  /// </summary>
  public double Priority;

  /// <summary>
  /// RootFlipDistance mode: estimated value change required at this node to reorder the root's
  /// move choice (NaN in VisitCount mode). Smaller is more decision-relevant.
  /// </summary>
  public double RequiredDelta = double.NaN;
}


/// <summary>
/// Target selection for PICKET M1 refutation grafting: a periodic depth-bounded region walk
/// from the search root (in the style of PrincipalRevaluation's region descent: visit cut
/// nCut = max(TargetNCutAbs, TargetCutFraction * rootN), onStack cycle guard, visited-set
/// DAG dedup) that collects probe-worthy interior nodes together with their root move paths.
///
/// Eligibility mirrors the DoSearchInnerNodes "rollable" check: evaluated, N &gt; 0,
/// non-terminal, NumPolicyMoves &gt; 0, and (unless TargetIncludeSearchRoot) not the search root.
/// Nodes probed too recently are skipped via a per-node ledger, re-probed only after N grows by
/// ReprobeNGrowthFactor - or by the smaller ReprobeNGrowthFactorShallow at or above
/// ReprobeShallowMaxDepth, since the geometric rule otherwise allows only O(log N) probes per node
/// and saturates coverage of the few high-leverage nodes near the root.
///
/// Priority follows TargetPriorityMode: either N gated on the dominant-child visit fraction
/// (VisitCount, the v1 rule), or proximity to changing the root's move (RootFlipDistance), which
/// ranks by the value change a candidate would need in order to reorder the root's children.
/// SEAM (proposal P6): targeting enrichment via QUnc / GNode.LeafValueVolatility signals
/// belongs here (a volatility term in the priority score), as a later, separately-tested
/// upgrade.
///
/// Runs only on the pump thread at the post-batch quiescent point (read-only graph access).
/// </summary>
internal sealed class ProbeGraftTargeter
{
  readonly ParamsProbeGraft Params;

  /// <summary>Per-node ledger: node N as of the last accepted probe submission.</summary>
  readonly Dictionary<int, int> lastProbeN = new();

  /// <summary>Scratch collections reused across sweeps.</summary>
  readonly List<ProbeGraftTargetSnapshot> candidates = new();
  readonly HashSet<int> visited = new();
  readonly HashSet<int> onStack = new();
  readonly List<MGMove> pathScratch = new();


  internal ProbeGraftTargeter(ParamsProbeGraft parms)
  {
    Params = parms;
  }



  #region Root flip distance

  /// <summary>Guards against division by zero for a candidate that is already at the flip point.</summary>
  const double REQUIRED_DELTA_EPSILON = 0.005;

  /// <summary>Priority discount applied when the dominant-visit gate does not hold.</summary>
  const double NON_DOMINANT_PRIORITY_DISCOUNT = 0.25;

  /// <summary>Per-slot gap (in Q units) between each root child and the best alternative root child.</summary>
  readonly List<double> rootChildGaps = new();


  /// <summary>
  /// Computes, for each expanded child slot of the search root, the Q gap that a refutation below
  /// it would have to overcome for the root to choose differently. Edge Q is from the child's
  /// perspective, so the root prefers the SMALLEST edge Q: for the current best child the relevant
  /// gap is to the runner-up (how much worse it would have to become), and for every other child
  /// it is the distance to the best child (how much better it would have to become). Both are
  /// non-negative.
  /// </summary>
  /// <param name="root"></param>
  void ComputeRootChildGaps(GNode root)
  {
    rootChildGaps.Clear();

    int numExpanded = root.NumEdgesExpanded;
    double bestQ = double.MaxValue;
    double secondQ = double.MaxValue;
    int bestSlot = -1;

    for (int slot = 0; slot < numExpanded; slot++)
    {
      GEdge edge = root.ChildEdgeAtIndex(slot);
      rootChildGaps.Add(0);

      if (edge.N <= 0)
      {
        continue;
      }

      double q = edge.Q;
      if (q < bestQ)
      {
        secondQ = bestQ;
        bestQ = q;
        bestSlot = slot;
      }
      else if (q < secondQ)
      {
        secondQ = q;
      }
    }

    if (bestSlot < 0)
    {
      return;
    }

    for (int slot = 0; slot < numExpanded; slot++)
    {
      GEdge edge = root.ChildEdgeAtIndex(slot);
      if (edge.N <= 0)
      {
        // Unvisited root children have no meaningful Q yet; treat them as maximally distant
        // rather than maximally urgent.
        rootChildGaps[slot] = double.MaxValue;
        continue;
      }

      rootChildGaps[slot] = slot == bestSlot
                          ? (secondQ == double.MaxValue ? double.MaxValue : secondQ - bestQ)
                          : edge.Q - bestQ;
    }
  }


  /// <summary>
  /// Gap for one root child slot (MaxValue when unknown, which suppresses targeting below it).
  /// </summary>
  /// <param name="slot"></param>
  double RootChildGap(int slot) => slot < rootChildGaps.Count ? rootChildGaps[slot] : double.MaxValue;


  /// <summary>
  /// Estimates the value change required at a candidate node for the root to choose a different
  /// move. A change of d at a node propagates to its root-child ancestor damped by that node's
  /// share of the ancestor's visits, so closing a gap g needs d = g * N(rootChild) / N(node).
  /// The search root itself is the decision, so it requires nothing.
  /// </summary>
  /// <param name="nodeN"></param>
  /// <param name="depth"></param>
  /// <param name="rootChildN"></param>
  /// <param name="rootChildGap"></param>
  static double RequiredDeltaToFlipRoot(int nodeN, int depth, int rootChildN, double rootChildGap)
  {
    if (depth == 0)
    {
      return 0;
    }

    if (rootChildGap == double.MaxValue || nodeN <= 0 || rootChildN <= 0)
    {
      return double.MaxValue;
    }

    return rootChildGap * rootChildN / Math.Max(1, nodeN);
  }

  #endregion

  /// <summary>
  /// Performs one target sweep over the current graph, returning the selected snapshots
  /// (top TargetMaxNodesPerSweep by priority, shallower depth breaking ties).
  /// </summary>
  /// <param name="engine"></param>
  /// <returns></returns>
  internal List<ProbeGraftTargetSnapshot> Sweep(MCGSEngine engine)
  {
    candidates.Clear();
    visited.Clear();
    onStack.Clear();
    pathScratch.Clear();

    GNode root = engine.SearchRootNode;
    int rootN = root.N;
    int nCut = Math.Max(Params.TargetNCutAbs, (int)(Params.TargetCutFraction * rootN));

    ComputeRootChildGaps(root);

    visited.Add(root.Index.Index);
    Walk(root, engine.SearchRootPosMG, 0, nCut, rootChildN: 0, rootChildGap: 0);

    candidates.Sort((a, b) => a.Priority != b.Priority ? b.Priority.CompareTo(a.Priority)
                                                       : a.Depth.CompareTo(b.Depth));
    if (candidates.Count > Params.TargetMaxNodesPerSweep)
    {
      candidates.RemoveRange(Params.TargetMaxNodesPerSweep, candidates.Count - Params.TargetMaxNodesPerSweep);
    }

    return candidates;
  }


  /// <summary>
  /// Records that a probe for the node was accepted by the source
  /// (arms the re-probe backoff at the node's current N).
  /// </summary>
  /// <param name="snapshot"></param>
  internal void RecordProbeSubmitted(ProbeGraftTargetSnapshot snapshot)
    => lastProbeN[snapshot.NodeIndex] = snapshot.N;


  /// <summary>
  /// Number of distinct graph nodes probed at least once during this search.
  /// </summary>
  internal int NumDistinctNodesProbed => lastProbeN.Count;


  /// <summary>
  /// Recursive descent below the given node (whose position is passed alongside so edge
  /// moves convert without CalcPosition walks). Children are descended only when above the
  /// visit cut; each qualifying non-root node is considered as a candidate once per sweep.
  /// </summary>
  void Walk(GNode node, in MGPosition nodePos, int depth, int nCut, int rootChildN, double rootChildGap)
  {
    if (depth > 0 || Params.TargetIncludeSearchRoot)
    {
      ConsiderCandidate(node, in nodePos, depth, rootChildN, rootChildGap);
    }

    if (depth >= Params.TargetMaxDepth)
    {
      return;
    }

    int nodeIdx = node.Index.Index;
    onStack.Add(nodeIdx);

    int numExpanded = node.NumEdgesExpanded;
    for (int slot = 0; slot < numExpanded; slot++)
    {
      GEdge edge = node.ChildEdgeAtIndex(slot);
      if (edge.Type != GEdgeStruct.EdgeType.ChildEdge || edge.ChildNodeIndex.IsNull)
      {
        continue;
      }

      GNode child = edge.ChildNode;
      int childIdx = child.Index.Index;
      if (child.N < nCut || onStack.Contains(childIdx) || !visited.Add(childIdx))
      {
        continue;
      }

      MGMove move = edge.MoveMGFromPos(in nodePos);
      MGPosition childPos = nodePos;
      childPos.MakeMove(move);

      // At depth 0 each child begins a new root-child subtree, whose visit count and
      // distance-to-best gap are carried down to every node below it.
      int childRootChildN = depth == 0 ? child.N : rootChildN;
      double childRootChildGap = depth == 0 ? RootChildGap(slot) : rootChildGap;

      pathScratch.Add(move);
      Walk(child, in childPos, depth + 1, nCut, childRootChildN, childRootChildGap);
      pathScratch.RemoveAt(pathScratch.Count - 1);
    }

    onStack.Remove(nodeIdx);
  }


  /// <summary>
  /// Evaluates one walked node for candidacy (eligibility, re-probe backoff) and
  /// records a snapshot when it qualifies.
  /// </summary>
  void ConsiderCandidate(GNode node, in MGPosition nodePos, int depth, int rootChildN, double rootChildGap)
  {
    bool eligible = node.IsEvaluated
                 && node.N > 0
                 && !node.Terminal.IsTerminal()
                 && node.NumPolicyMoves > 0
                 && (!node.IsSearchRoot || Params.TargetIncludeSearchRoot);
    if (!eligible)
    {
      return;
    }

    // Nodes near the root carry the most decision leverage, so they use the (smaller) shallow
    // re-probe spacing: the geometric rule alone allows only O(log N) probes per node, which
    // saturates their coverage long before probe capacity is exhausted.
    float growthFactorRequired = depth <= Params.ReprobeShallowMaxDepth
                               ? Math.Min(Params.ReprobeNGrowthFactorShallow, Params.ReprobeNGrowthFactor)
                               : Params.ReprobeNGrowthFactor;

    int nodeIdx = node.Index.Index;
    if (lastProbeN.TryGetValue(nodeIdx, out int probedAtN)
     && node.N < (long)(probedAtN * growthFactorRequired))
    {
      return; // probed too recently relative to visit growth
    }

    // Dominant child by visit count (over expanded edges; unexpanded children have N = 0).
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

    float dominantFraction = dominantIndex < 0 ? 0 : (float)dominantN / node.N;
    bool gateHeld = dominantFraction >= Params.DominantChildMinVisitFraction;

    double requiredDelta = double.NaN;
    double priority;
    if (Params.TargetPriorityMode == ParamsProbeGraft.TargetPriorityModeType.RootFlipDistance)
    {
      requiredDelta = RequiredDeltaToFlipRoot(node.N, depth, rootChildN, rootChildGap);
      if (requiredDelta > Params.TargetMaxRequiredDelta)
      {
        return; // no plausible refutation here could change the move played
      }

      // Rank by ascending required change; the dominant-visit gate still favors nodes where a
      // refutation trigger (as opposed to a discovery trigger) is possible at all.
      priority = 1.0 / (requiredDelta + REQUIRED_DELTA_EPSILON);
      if (!gateHeld)
      {
        priority *= NON_DOMINANT_PRIORITY_DISCOUNT;
      }
    }
    else
    {
      priority = gateHeld ? node.N : 0;
    }

    ProbeGraftTargetSnapshot snapshot = new()
    {
      NodeIndex = nodeIdx,
      N = node.N,
      Q = node.Q,
      Depth = depth,
      DominantChildIndex = dominantIndex,
      DominantVisitFraction = dominantFraction,
      DominantMove = dominantIndex < 0 ? default
                                       : node.ChildEdgeAtIndex(dominantIndex).MoveMGFromPos(in nodePos),
      DominantFractionGateHeld = gateHeld,
      MovesFromRoot = pathScratch.ToArray(),
      Priority = priority,
      RequiredDelta = requiredDelta,
    };

    candidates.Add(snapshot);
  }
}
