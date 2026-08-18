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
using System.Diagnostics;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Text;

using Ceres.Base.DataType;
using Ceres.Base.Math;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Search.PUCT;

/// <summary>
/// SIMD (AVX) code for selecting one or (typically) more children
/// to be followed next according to PUCT in the tree descent,
/// using a formula to balance exploration vs. exploitation.
/// 
/// Note that AVX512 is not currently automatically used by System.Numerics.Vector even in .NET 10:
/// "We want to make the default 512 bit in the future..."
/// though an environment variable may enable this:
///   set DOTNET_MaxVectorTBitWidth = 512
///
/// The per-child math runs in float rather than double. This doubles the number of children
/// processed per SIMD instruction (8 vs 4 lanes at 256 bits) and avoids the substantially weaker
/// double throughput of ARM64 NEON; measured 1.25x (4 children) to 2.1x (64 children) faster on
/// x64. Float's ~7 significant digits leave the selected child unchanged: score differences
/// against the double implementation are ~1e-7, far below any margin that decides selection
/// (and exact ties, which become slightly more common, resolve to the lowest index exactly as
/// before - see ArrayUtils.IndexOfElementWithMaxValue). Scalar quantities that accumulate over
/// the whole node (parent N and in-flight counts, the CPUCT product) remain double and are
/// narrowed only when entering the vector kernel.
/// </summary>
public unsafe static class PUCTScoreCalcVector
{
  public const int MAX_CHILDREN = 64;


  /// <summary>
  /// Static variable available for debugging purposes to
  /// control of SIMD versions of PUCT logic are used
  /// instead of C# fallback.
  /// </summary>
  public static bool ENABLE_SIMD_CALCS = true;


  /// <summary>
  /// Entry point for the score calculation.
  /// 
  /// Given a set of information about the current node 
  /// the number of visits to be made to the subtree, returns:
  ///   - a Span of visit counts indicating the number of visits warranted for each child
  ///   - optionally a Span of the computed child PUCT scores
  /// </summary>
  /// <param name="paramsSelect"></param>
  /// <param name="parentIsRoot"></param>
  /// <param name="parentN"></param>
  /// <param name="parentNInFlight"></param>
  /// <param name="qParent"></param>
  /// <param name="parentSumPVisited"></param>
  /// <param name="childStats"></param>
  /// <param name="qWhenNoChildrenPerChild"></param>
  /// <param name="numChildren"></param>
  /// <param name="numVisitsToCompute"></param>
  /// <param name="outputScores"></param>
  /// <param name="outputChildVisitCounts"></param>
  /// <param name="cpuctMultiplier"></param>
  /// <param name="thresholdPUCTSuboptimalityReject"></param>
  /// <param name="parentNode"></param>
  /// <param name="quncScoreBonus">Optional per-child additive score adjustment (Q-uncertainty
  /// methods M1/M2/M3). Null = exact stock behavior. When non-null, must extend at least
  /// Vector&lt;float&gt;.Count entries past numChildren with zero fill (SIMD block loads).</param>
  /// <param name="quncUMultiplier">Optional per-child multiplier on the U (exploration) term
  /// (Q-uncertainty method M4). Null = exact stock behavior. Same padding contract, fill 1.</param>
  /// <returns></returns>
  internal static int ScoreCalcMulti(ParamsSelect paramsSelect,
                                     bool parentIsRoot, int parentN, double parentNInFlight,
                                     double qParent, double parentSumPVisited,
                                     GatheredChildStats childStats,
                                     float[] qWhenNoChildrenPerChild,
                                     int numChildren, int numVisitsToCompute,
                                     Span<double> outputScores, Span<short> outputChildVisitCounts,
                                     double cpuctMultiplier,
                                     float thresholdPUCTSuboptimalityReject,
                                     GNode parentNode = default,
                                     float[] quncScoreBonus = null,
                                     float[] quncUMultiplier = null)
  {
    Debug.Assert(!double.IsNaN(qParent));

    // Saving output scores only makes sense when a single visit being computed
    Debug.Assert(!(!outputScores.IsEmpty && numVisitsToCompute > 1));
    Debug.Assert(numChildren <= MAX_CHILDREN);

    Debug.Assert(outputScores.IsEmpty || outputScores.Length >= numChildren);
    Debug.Assert(numVisitsToCompute == 0 || outputChildVisitCounts.Length >= numChildren);

    float virtualLossMultiplier;
    if (ParamsSelect.VLossRelative)
    {
      virtualLossMultiplier = (float)qParent + paramsSelect.VirtualLossDefaultRelative;
    }
    else
    {
      virtualLossMultiplier = paramsSelect.VirtualLossDefaultAbsolute;
    }

    double cpuctValue = cpuctMultiplier * paramsSelect.CalcCPUCT(parentIsRoot, parentN);

    // Compute qWhenNoChildren based on FPU mode
    // TODO: to be more precise, parentSumPVisited should possibly be updated as we visit children
    double qWhenNoChildren = paramsSelect.CalcQWhenNoChildren(parentIsRoot, qParent, parentSumPVisited);

    const bool DUMP_Q_WHEN_NO_CHILDREN = false;
    if (DUMP_Q_WHEN_NO_CHILDREN && qWhenNoChildrenPerChild != null)
    {
      Console.WriteLine($"\r\nN= {parentN} Q={qParent}");
      for (int i = 0; i < numChildren; i++)
      {
        double delta = qWhenNoChildrenPerChild[i] - qWhenNoChildren;
        Console.WriteLine($"{100 * childStats.P.Span[i],6:F0}%  {(childStats.W.Span[i] / childStats.N.Span[i]),6:F2}  DELTA= {delta,6:F2}  {qWhenNoChildrenPerChild[i],6:F2}  was: {qWhenNoChildren,6:F2}");
      }
    }

    if (parentIsRoot
      && parentN > paramsSelect.RootCPUCTExtraMultiplierDivisor
      && paramsSelect.RootCPUCTExtraMultiplierExponent != 0)
    {
      cpuctValue *= Math.Pow(parentN / paramsSelect.RootCPUCTExtraMultiplierDivisor,
                              paramsSelect.RootCPUCTExtraMultiplierExponent);
    }

    int numVisitsAccepted = Compute(parentN, qParent, parentNInFlight, childStats, numChildren, numVisitsToCompute, outputScores,
                                    outputChildVisitCounts, virtualLossMultiplier,
                                    parentIsRoot ? paramsSelect.UCTRootNumeratorExponent : paramsSelect.UCTNonRootNumeratorExponent,
                                    cpuctValue, qWhenNoChildren, qWhenNoChildrenPerChild,
                                    parentIsRoot ? paramsSelect.UCTRootDenominatorExponent : paramsSelect.UCTNonRootDenominatorExponent,
                                    thresholdPUCTSuboptimalityReject,
                                    quncScoreBonus, quncUMultiplier);
    return numVisitsAccepted;
  }



  /// <summary>
  /// Worker method that coordinates looping over all the requested visits,
  /// including a performance optimization that attempts to 
  /// detect the condition where many consecutive visits will be made to the same child.
  /// </summary>
  /// <param name="parentN"></param>
  /// <param name="parentNInFlight"></param>
  /// <param name="p"></param>
  /// <param name="w"></param>
  /// <param name="n"></param>
  /// <param name="nInFlight"></param>
  /// <param name="numChildren"></param>
  /// <param name="numVisitsToCompute"></param>
  /// <param name="outputScores"></param>
  /// <param name="outputChildVisitCounts"></param>
  /// <param name="numBlocks"></param>
  /// <param name="virtualLossMultiplier"></param>
  /// <param name="uctParentPower"></param>
  /// <param name="cpuctValue"></param>
  /// <param name="qWhenNoChildren"></param>
  /// <param name="uctDenominatorPower"></param>
  [SkipLocalsInit]
  private static int Compute(int parentN, double qParent, double parentNInFlight,
                             GatheredChildStats childStats,
                             int numChildren, int numVisitsToCompute,
                             Span<double> outputScores, Span<short> outputChildVisitCounts,
                             float virtualLossMultiplier, float uctParentPower,
                             double cpuctValue,
                             double qWhenNoChildren, float[] qWhenNoChildrenPerChild,
                             double uctDenominatorPower,
                             float thresholdPUCTSuboptimalityReject,
                             float[] quncScoreBonus = null, float[] quncUMultiplier = null)
  {
    // Load the vectors that do not change
    Span<float> nInFlight = childStats.NInFlightAdjusted.Span;


    int maxScratchChildren = (int)MathUtils.RoundedUp(Math.Min(MAX_CHILDREN, numChildren + numVisitsToCompute), Vector<float>.Count);
    Span<float> childScores = stackalloc float[maxScratchChildren];

    int numVisits = 0;
    int numTooSuboptimal = 0;

    // Allow at least 1 and up to 20% of requested visits to exceed the limit (or up to 5% of parent N).
    int maxVisitsAllowedOverSuboptimalityLimit = 1 + Math.Max(numVisitsToCompute / 5, parentN / 20);

    while (numVisits < numVisitsToCompute
        || numVisits == 0 && numVisitsToCompute == 0) // just querying scores, no children to select
    {
      // Get constant term handy
      double numVisitsByParentToChildren = parentNInFlight + (parentN < 2 ? 1 : parentN - 1);
      double cpuctSqrtParentN = cpuctValue * ParamsSelect.UCTParentMultiplier(numVisitsByParentToChildren, uctParentPower);
      ComputeChildScores(childStats, numChildren, qWhenNoChildren, qWhenNoChildrenPerChild, virtualLossMultiplier,
                         childScores, cpuctSqrtParentN, uctDenominatorPower, quncScoreBonus, quncUMultiplier);

      // Assert that none of the scores were NaN.
#if DEBUG
      // TODO: remove conditional compilation once TensorPrimitives versioning issue resolved.
      Debug.Assert(!float.IsNaN(TensorPrimitives.Max(childScores[..numChildren])));
#endif
      // Save back to output scores (if these were requested)
      if (!outputScores.IsEmpty)
      {
        Debug.Assert(numVisits <= 1);

        // Scores are exposed to callers (diagnostics / root move statistics) as double;
        // widen on the way out rather than holding the whole kernel at double precision.
        Span<float> scoresSpan = childScores[..numChildren];
        for (int i = 0; i < scoresSpan.Length; i++)
        {
          outputScores[i] = scoresSpan[i];
        }
      }

      if (numVisitsToCompute == 0)
      {
        return numVisits;
      }

      // Find the best child and record this visit
      int maxIndex = ArrayUtils.IndexOfElementWithMaxValue(childScores, numChildren);

      if (thresholdPUCTSuboptimalityReject < float.MaxValue)
      {
        // Determine suboptimality of this child wrt. parent Q.
        double thisQ = childStats.W.Span[maxIndex] / childStats.N.Span[maxIndex];
        double qSuboptimality = thisQ + qParent;

        if (qSuboptimality > thresholdPUCTSuboptimalityReject)
        {
          numTooSuboptimal++;

          if (numTooSuboptimal > maxVisitsAllowedOverSuboptimalityLimit)
          {
            //if (Random.Shared.Next(100) == 0)
            //  Console.WriteLine($"Reducing visit count from {numVisitsToCompute} to {numVisits} at parentN= {parentN} due to Q suboptimality {qSuboptimality}");
            return numVisits;
          }
        }
      }

      // Update items to reflect this visit
      parentNInFlight += 1;
      nInFlight[maxIndex] += 1;
      numVisits += 1;

      outputChildVisitCounts[maxIndex] += 1;

      int numRemainingVisits = numVisitsToCompute - numVisits;

      // If we just found our first child we repeatedly try to  
      // "jump ahead" by 10 visits at a time 
      // as long as the first top child child remains the best child.
      // This optimizes for the common case that one child is dominant,
      // and empirically reduces the number of calls to ComputeChildScores by more than 30%. 
      const int REPEATED_VISITS_DIVISOR = 10;
      int numAdditionalTryVisitsPerIteration = Math.Max(10, numRemainingVisits / REPEATED_VISITS_DIVISOR);
      if (numVisits == 1 && numRemainingVisits > numAdditionalTryVisitsPerIteration + 5)
      {
        int numSuccessfulVisitsAllIterations = 0;

        do
        {
          // Modify state to simulate additional visits to this top child
          double newNInFlight = nInFlight[maxIndex] += numAdditionalTryVisitsPerIteration;

          // Compute new child scores
          numVisitsByParentToChildren = newNInFlight + parentNInFlight + (parentN < 2 ? 1 : parentN - 1);
          cpuctSqrtParentN = cpuctValue * ParamsSelect.UCTParentMultiplier(numVisitsByParentToChildren, uctParentPower);
          ComputeChildScores(childStats, numChildren, qWhenNoChildren, qWhenNoChildrenPerChild, virtualLossMultiplier,
                             childScores, cpuctSqrtParentN, uctDenominatorPower, quncScoreBonus, quncUMultiplier);

          // Check if the best child was still the same
          if (maxIndex == ArrayUtils.IndexOfElementWithMaxValue(childScores, numChildren))
          {
            // Child remained same, increment successful count
            numSuccessfulVisitsAllIterations += numAdditionalTryVisitsPerIteration;
          }
          else
          {
            // Failed, back out the last update to nInFlight and stop iterating
            nInFlight[maxIndex] -= numAdditionalTryVisitsPerIteration;

            break;
          }

          numAdditionalTryVisitsPerIteration = Math.Max(10, numRemainingVisits / REPEATED_VISITS_DIVISOR);
        } while (numRemainingVisits - numSuccessfulVisitsAllIterations > numAdditionalTryVisitsPerIteration);

        if (numSuccessfulVisitsAllIterations > 0)
        {
          // The nInFlight have already been kept continuously up to date
          // but need to update the other items to reflect these visits
          parentNInFlight += numSuccessfulVisitsAllIterations;
          numVisits += numSuccessfulVisitsAllIterations;
          if (!outputChildVisitCounts.IsEmpty)
          {
            outputChildVisitCounts[maxIndex] += (short)numSuccessfulVisitsAllIterations;
          }
        }
      }
    }

    return numVisits;
  }


  /// <summary>
  /// Computes the PUCT child scores for this node into computedChildScores.
  /// </summary>
  /// <param name="p"></param>
  /// <param name="w"></param>
  /// <param name="n"></param>
  /// <param name="nInFlight"></param>
  /// <param name="numChildren"></param>
  /// <param name="qWhenNoChildren"></param>
  /// <param name="virtualLossMultiplier"></param>
  /// <param name="computedChildScores"></param>
  /// <param name="cpuctSqrtParentN"></param>
  /// <param name="uctDenominatorPower"></param>
  private static void ComputeChildScores(GatheredChildStats childStats,
                                         int numChildren,
                                         double qWhenNoChildren, float[] qWhenNoChildrenPerChild,
                                         double virtualLossMultiplier, Span<float> computedChildScores,
                                         double cpuctSqrtParentN, double uctDenominatorPower,
                                         float[] quncScoreBonus = null, float[] quncUMultiplier = null)
  {
    // Note: SIMD path blends action into Q globally via weight (Q = (1-w)*Q + w*A).
    //       The new FPUType.ActionHead mode uses action values as per-child FPU instead,
    //       which flows through qWhenNoChildrenPerChild and does not require the blending logic.

#if OLD_ACTION_COMMENT
    Need to review / harmonize logic between these two methods(pick which one is intended).
Findings comparing ComputeChildScoresSIMD vs ComputeChildScoresNonSIMD(ignoring commented -out code):
	Action - head logic not equivalent(High)
	SIMD path: Only blends action head into Q if ACTION_ENABLED is defined, 
  using Q = (1 - w)*Q + w*A for all items when weight != 0.
	Non - SIMD path: Always compiled and applies a different rule
  only for unvisited moves (i > 0 && N[i] == 0 && weight != 0): 
    Q = max(Q, A[i] + 0.10).No global blending.
This changes selection behavior even when ACTION_ENABLED is not defined and adds a fixed +0.10 offset floor.
#endif

    if (ENABLE_SIMD_CALCS && Vector.IsHardwareAccelerated)
    {
      ComputeChildScoresSIMD(childStats, numChildren, qWhenNoChildren, qWhenNoChildrenPerChild,
                             virtualLossMultiplier, computedChildScores,
                             cpuctSqrtParentN, uctDenominatorPower, quncScoreBonus, quncUMultiplier);
    }
    else
    {
      ComputeChildScoresNonSIMD(childStats, numChildren, qWhenNoChildren, qWhenNoChildrenPerChild,
                                virtualLossMultiplier, computedChildScores,
                                cpuctSqrtParentN, uctDenominatorPower, quncScoreBonus, quncUMultiplier);
    }
  }


  private static void ComputeChildScoresSIMD(GatheredChildStats childStats,
                                             int numChildren,
                                             double qWhenNoChildren, float[] qWhenNoChildrenPerChild,
                                             double virtualLossMultiplier, Span<float> computedChildScores,
                                             double cpuctSqrtParentN, double uctDenominatorPower,
                                             float[] quncScoreBonus = null, float[] quncUMultiplier = null)
  {
    // Narrow the node-level scalars once, outside the per-block loop.
    float virtualLossMultiplierF = (float)virtualLossMultiplier;
    float cpuctSqrtParentNF = (float)cpuctSqrtParentN;
    float qWhenNoChildrenF = (float)qWhenNoChildren;

    int simdWidth = Vector<float>.Count;
    int numBlocks = numChildren / simdWidth + (numChildren % simdWidth == 0 ? 0 : 1);

    Span<float> p = childStats.P.Span;
    Span<float> w = childStats.W.Span;
    Span<float> n = childStats.N.Span;
#if ACTION_ENABLED
    Span<float> a = childStats.A.Span;
#endif
    Span<float> nInFlight = childStats.NInFlightAdjusted.Span;

    int blockCount = 0;
    while (blockCount < numBlocks)
    {
      int startOffset = blockCount * simdWidth;

      // Load vectors from spans (caller guarantees adequate padding; no tail handling required)
      Vector<float> vW = new(w[startOffset..]);
      Vector<float> vN = new(n[startOffset..]);
      Vector<float> vP = new(p[startOffset..]);
#if ACTION_ENABLED
      Vector<float> vA = new Vector<float>(a[startOffset..]);
#endif
      Vector<float> vQWhenNoChildren;
      if (qWhenNoChildrenPerChild != null)
      {
        ReadOnlySpan<float> qNoChildrenSpan = qWhenNoChildrenPerChild.AsSpan(startOffset);

        if (qNoChildrenSpan.Length >= Vector<float>.Count)
        {
          // Hot path: a full vector is available - load straight from the array.
          vQWhenNoChildren = new Vector<float>(qNoChildrenSpan);
        }
        else
        {
          // Tail only: pad the missing lanes with the scalar fallback.
          Span<float> padded = stackalloc float[Vector<float>.Count];
          padded.Fill(qWhenNoChildrenF);
          qNoChildrenSpan.CopyTo(padded);
          vQWhenNoChildren = new Vector<float>(padded);
        }
      }
      else
      {
        vQWhenNoChildren = new Vector<float>(qWhenNoChildrenF);
      }
      Vector<float> vNInFlight = new(nInFlight[startOffset..]);

      Vector<float> vScore;
      if (quncScoreBonus == null && quncUMultiplier == null)
      {
        // Stock path: byte-identical kernel when the Q-uncertainty methods are inactive.
        vScore = ComputeScoresSIMD(vW, vN, vP,
#if ACTION_ENABLED
                                   vA,
#endif
                                   virtualLossMultiplierF, cpuctSqrtParentNF, uctDenominatorPower,
                                   vQWhenNoChildren, vNInFlight);
      }
      else
      {
        // Adjustment arrays are guaranteed by the caller to extend a full vector past
        // numChildren with neutral fill (0 bonus / 1 multiplier).
        Vector<float> vBonus = quncScoreBonus != null
            ? new Vector<float>(quncScoreBonus.AsSpan(startOffset)) : Vector<float>.Zero;
        Vector<float> vUMult = quncUMultiplier != null
            ? new Vector<float>(quncUMultiplier.AsSpan(startOffset)) : Vector<float>.One;
        vScore = ComputeScoresSIMDQUnc(vW, vN, vP,
#if ACTION_ENABLED
                                       vA,
#endif
                                       virtualLossMultiplierF, cpuctSqrtParentNF, uctDenominatorPower,
                                       vQWhenNoChildren, vNInFlight, vUMult, vBonus);
      }

      vScore.CopyTo(computedChildScores[startOffset..]);

      blockCount++;
    }
  }



#if FEATURE_UNCERTAINTY_SCALING
      const double AVG_UV = 10;
      const double POW = 0.5; // if not 0.5, need to use ToPowerAVX below
      const double MULTIPLIER = 0.15;

      // Take sqrt(uv) and approximately center in a range approximately [-3, +3]
      Vector256<double> vUAdj = Avx.Sqrt(vUV); // ** NOTE: Must use POW=0.5 above!
      //Vector256<double> vUAdj = ToPowerAVX(vUV,POW);
      vUAdj = Avx.Subtract(vUAdj, Vector256.Create(MathF.Pow(AVG_UV, POW)));

      // Replace elements in vuADJ with 0 if the corresponding element in nPlusNInFlightPlus1 is identically zero
      Vector256<double> mask = Avx.Compare(vNPlusNInFlight, Vector256.Create(0f), FloatComparisonMode.OrderedEqualNonSignaling);
      vUAdj = Avx.AndNot(mask, vUAdj);

      if (false) 
      {
        // Replace uncertainty adjustment with 0 for nodes which are better than parent
        Vector256<double> v = Avx.Divide(vW, vN);
        Vector256<double> mask1 = Avx.Compare(v, Vector256.Create(-qParent), FloatComparisonMode.OrderedLessThanNonSignaling);
        vUAdj = Avx.AndNot(mask1, vUAdj);
      }

      // Scale down adjustment by some factor
      vUAdj = Avx.Multiply(vUAdj, Vector256.Create(MULTIPLIER));

      // Divide adjustment by sqrt(N+NInFlight+1) to reduce influence as more visits are made
      Vector256<double> nPlusNInFlightPlus1 = Avx.Add(vNPlusNInFlight, Vector256.Create(1f));

      // Reduce magnitude of adjustment as more visits are made
      vUAdj = Avx.Divide(vUAdj, nPlusNInFlightPlus1); // divide by number of visits already made

      // Finally, center around 1 instead of 0
      vUAdj = Avx.Add(vUAdj, Vector256.Create(1f));
      denominator = Avx.Divide(denominator, vUAdj);
    }
#endif

  #region Non-SIMD versions

  /// <summary>
  /// Direct C# version of ComputeChildScores without use of SIMD.
  /// About 60% as fast as the AVX version.
  /// </summary>
  private unsafe static void ComputeChildScoresNonSIMD(GatheredChildStats childStats,
                                                       int numChildren,
                                                       double qWhenNoChildren, float[] qWhenNoChildrenPerChild,
                                                       double virtualLossMultiplier, Span<float> computedChildScores,
                                                       double cpuctSqrtParentN, double uctDenominatorPower,
                                                       float[] quncScoreBonus = null, float[] quncUMultiplier = null)
  {
    ComputeScoresNonSIMD(numChildren, childStats.W.Span, childStats.N.Span,
                         childStats.P.Span, childStats.A.Span,
                         virtualLossMultiplier, cpuctSqrtParentN, uctDenominatorPower,
                         qWhenNoChildren, qWhenNoChildrenPerChild,
                         childStats.NInFlightAdjusted.Span, computedChildScores,
                         quncScoreBonus, quncUMultiplier);
  }


  private static void ComputeScoresNonSIMD(int numScores, Span<float> vW, Span<float> vN, Span<float> vP, Span<float> vA,
                                           double virtualLossMultiplier,
                                           double cpuctSqrtParentN,
                                           double uctDenominatorPower,
                                           double qWhenNoChildren, float[] qWhenNoChildrenPerChild,
                                           Span<float> vNInFlight,
                                           Span<float> outputVScore,
                                           float[] quncScoreBonus = null, float[] quncUMultiplier = null)
  {
    // Kept in float to mirror the SIMD kernel exactly (this path is a debug/fallback alternative
    // to it, selected by ENABLE_SIMD_CALCS or absence of hardware acceleration).
    float virtualLossMultiplierF = (float)virtualLossMultiplier;
    float cpuctSqrtParentNF = (float)cpuctSqrtParentN;
    float qWhenNoChildrenF = (float)qWhenNoChildren;

    for (int i = 0; i < numScores; i++)
    {
      float nPlusNInFlight = vN[i] + vNInFlight[i];
      float _denominator;
      if (uctDenominatorPower == 1.0f)
      {
        _denominator = nPlusNInFlight;
      }
      else if (uctDenominatorPower == 0.5f)
      {
        _denominator = MathF.Sqrt(nPlusNInFlight);
      }
      else
      {
        _denominator = MathF.Pow(nPlusNInFlight, (float)uctDenominatorPower);
      }

      float _vQ;
      float _vLossContrib = vNInFlight[i] * virtualLossMultiplierF;
      if (nPlusNInFlight > 0)
      {
        _vQ = (_vLossContrib - vW[i]) / nPlusNInFlight;
      }
      else
      {
        float thisQWhenNoChildren = qWhenNoChildrenPerChild != null ? qWhenNoChildrenPerChild[i] : qWhenNoChildrenF;
        _vQ = thisQWhenNoChildren + _vLossContrib;
      }

      // [Experimental action blending, superseded by FPUType.ActionHead mode]
      // if (i > 0 && vN[i] == 0 && actionHeadSelectionWeight != 0)
      // {
      //   double aDiff = vA[i] - vA[0];
      //   _vQ = Math.Max(_vQ, vA[i] + 0.10f);
      // }

      // U
      float _vUNumerator = vP[i] * cpuctSqrtParentNF;
      float _vDenominator = 1 + _denominator;
      float _vU = _vUNumerator / _vDenominator;

      // Optional Q-uncertainty adjustments (M4 multiplies U; M1/M2/M3 add to the score).
      if (quncUMultiplier != null)
      {
        _vU *= quncUMultiplier[i];
      }
      float _vScore = _vU + _vQ;
      if (quncScoreBonus != null)
      {
        _vScore += quncScoreBonus[i];
      }

      outputVScore[i] = _vScore;
    }
  }

  #endregion


  #region Platform-agnostic

  /// <summary>
  /// Platform-agnostic vectorized worker that implements the CPUCT math using System.Numerics.Vector.
  /// The JIT maps this to AVX/AVX2 on x86 and AdvSimd on ARM64. It does not currently auto-use AVX-512 for Vector<T>.
  /// </summary>
  /// <remarks>
  /// Caller guarantees input spans are sized in multiples of Vector<float>.Count, so no tail handling here.
  /// </remarks>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static Vector<float> ComputeScoresSIMD(Vector<float> vW, Vector<float> vN, Vector<float> vP,
#if ACTION_ENABLED
                                                 Vector<float> vA,
#endif
                                                 float virtualLossMultiplier,
                                                 float cpuctSqrtParentN, double uctDenominatorPower,
                                                 Vector<float> vQWhenNoChildren, Vector<float> vNInFlight)
  {
    Vector<float> vNPlusNInFlight = vN + vNInFlight;
    Vector<float> vVirtualLossMultiplier = new Vector<float>(virtualLossMultiplier);

    Vector<float> denominator;
    if (uctDenominatorPower == 1.0)
    {
      denominator = vNPlusNInFlight;
    }
    else if (uctDenominatorPower == 0.5)
    {
      denominator = Vector.SquareRoot(vNPlusNInFlight);
    }
    else
    {
      denominator = ToPowerVector(vNPlusNInFlight, uctDenominatorPower);
    }

    Vector<float> vLossContrib = vNInFlight * vVirtualLossMultiplier;

    Vector<float> vCPUCTSqrtParentN = new(cpuctSqrtParentN);
    Vector<float> vUNumerator = vP * vCPUCTSqrtParentN;
    Vector<float> vDenominator = Vector<float>.One + denominator;
    Vector<float> vU = vUNumerator / vDenominator;

    Vector<float> vQWithChildren = (vLossContrib - vW) / vNPlusNInFlight;
    Vector<float> vQWithoutChildren = vQWhenNoChildren + vLossContrib;
    Vector<int> maskNoChildren = Vector.GreaterThan(vNPlusNInFlight, Vector<float>.Zero);
    Vector<float> vQ = Vector.ConditionalSelect(maskNoChildren, vQWithChildren, vQWithoutChildren);

    Vector<float> vScore = vU + vQ;
    return vScore;
  }


  /// <summary>
  /// Variant of ComputeScoresSIMD applying the optional Q-uncertainty adjustments
  /// (final score = U * uMultiplier + Q + scoreBonus). Kept as a DUPLICATE of the
  /// stock kernel (rather than folding neutral adjustments into it) so the inactive
  /// path stays byte-identical - the Phase 2 invariance gate is structural.
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static Vector<float> ComputeScoresSIMDQUnc(Vector<float> vW, Vector<float> vN, Vector<float> vP,
#if ACTION_ENABLED
                                                      Vector<float> vA,
#endif
                                                      float virtualLossMultiplier,
                                                      float cpuctSqrtParentN, double uctDenominatorPower,
                                                      Vector<float> vQWhenNoChildren, Vector<float> vNInFlight,
                                                      Vector<float> vUMultiplier, Vector<float> vScoreBonus)
  {
    Vector<float> vNPlusNInFlight = vN + vNInFlight;
    Vector<float> vVirtualLossMultiplier = new Vector<float>(virtualLossMultiplier);

    Vector<float> denominator;
    if (uctDenominatorPower == 1.0)
    {
      denominator = vNPlusNInFlight;
    }
    else if (uctDenominatorPower == 0.5)
    {
      denominator = Vector.SquareRoot(vNPlusNInFlight);
    }
    else
    {
      denominator = ToPowerVector(vNPlusNInFlight, uctDenominatorPower);
    }

    Vector<float> vLossContrib = vNInFlight * vVirtualLossMultiplier;

    Vector<float> vCPUCTSqrtParentN = new(cpuctSqrtParentN);
    Vector<float> vUNumerator = vP * vCPUCTSqrtParentN;
    Vector<float> vDenominator = Vector<float>.One + denominator;
    Vector<float> vU = vUNumerator / vDenominator;

    Vector<float> vQWithChildren = (vLossContrib - vW) / vNPlusNInFlight;
    Vector<float> vQWithoutChildren = vQWhenNoChildren + vLossContrib;
    Vector<int> maskNoChildren = Vector.GreaterThan(vNPlusNInFlight, Vector<float>.Zero);
    Vector<float> vQ = Vector.ConditionalSelect(maskNoChildren, vQWithChildren, vQWithoutChildren);

    Vector<float> vScore = vU * vUMultiplier + vQ + vScoreBonus;
    return vScore;
  }


  // Platform-agnostic power for System.Numerics.Vector<float>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  static Vector<float> ToPowerVector(Vector<float> values, double power)
  {
    Span<float> buf = stackalloc float[Vector<float>.Count];
    values.CopyTo(buf);
    float powerF = (float)power;
    for (int i = 0; i < buf.Length; i++)
    {
      buf[i] = MathF.Pow(buf[i], powerF);
    }
    return new Vector<float>(buf);
  }

  #endregion

}
