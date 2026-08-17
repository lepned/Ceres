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

using Ceres.Chess;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Managers.Limits;

/// <summary>
/// Testbed limits manager (selected via Ceres.json "LimitsManagerName": "TEST"):
/// a clock-trajectory controller, offered as a structurally simpler alternative to the
/// multiplicative-factor scheme of ManagerGameLimitCeresMCGS.
///
/// Principle: close the loop on the observable clock state rather than on predicted
/// consumption. At the first move of a game a target remaining-clock schedule R*(m) is
/// committed, gliding from the starting clock down to a small terminal reserve over the
/// expected number of own moves. Each move then allocates
///
///   alloc = 0.98*increment + Aggressiveness * (drain + GAIN * (R - R*(m))) * winFactor
///
/// where drain = (R - reserve) * SHAPE_P / E[own moves left] is the planned glide step
/// using a LIVE horizon estimate, and the error term redeploys any surplus versus the
/// committed schedule. Time handed back by early search termination (futility stop) simply
/// leaves the clock ahead of schedule and is automatically re-spent at rate GAIN — no
/// consumption model or feedback ratio is needed, and the variability of that giveback
/// (opponent similarity, tree reuse, time control) is absorbed by construction.
///
/// The horizon estimate is deliberately NN-free (no reliance on a moves-left head, which is
/// often absent and of poor quality): a piece-count base shortened smoothly as |rootQ| grows,
/// since decisive positions end games early (by adjudication or resignation-like collapses) —
/// exactly the games where end-of-game leftover otherwise concentrates. A short horizon
/// raises the per-move drain, accelerating spending as the game approaches its end.
///
/// The mean final leftover is governed directly by the terminal reserve (plus horizon
/// estimation error) instead of emerging from the equilibrium of many multiplied factors.
/// Long games park at the reserve, spending about one increment per move.
///
/// Safety envelope (identical in spirit to the default manager): panic/near-exhaustion
/// short-circuits, never more than MAX_FRACTION_REMAINING_PER_MOVE of the remaining clock,
/// and a minimum allocation floor. Units are agnostic: works for per-game seconds or nodes.
/// </summary>
[Serializable]
public class ManagerGameLimitTest : IManagerGameLimit
{
  /// <summary>
  /// Scales the controller terms (drain + error correction), from
  /// ParamsSearch.GameLimitUsageAggressiveness. Constants below were tuned at 1.0.
  /// </summary>
  public readonly float Aggressiveness;

  /// <summary>
  /// If set, every ComputeMoveAllocation builds a diagnostic string (attached to the outputs)
  /// describing how the allocation was derived, so it can be recorded in the game move-log.
  /// </summary>
  public bool CaptureDiagnostics { get; set; }

  // Front-load exponent of the committed schedule (1 = linear glide; >1 spends more early).
  const float SHAPE_P = 2.2f;

  // Fraction of the surplus versus the committed schedule re-spent per move.
  const float ERROR_GAIN = 0.20f;

  // Terminal reserve the schedule glides toward: max of these two expressions.
  const float RESERVE_MIN_INCREMENTS = 1.5f;
  const float RESERVE_FRACTION_OF_START = 0.03f;

  // Extra planned drain on the first move of a game (initial tree build, no reuse).
  const float FIRST_MOVE_FACTOR = 2.0f;

  // Fraction of the per-move increment always spent (as in the default manager).
  const float FRACTION_OF_INCREMENT = 0.98f;

  // Anti-flag envelope, identical to the default manager.
  const float MAX_FRACTION_REMAINING_PER_MOVE = 0.50f;
  const float MIN_TIME = 0.05f;

  // Live horizon (expected own moves remaining) comes from the shared NN-free estimator
  // (piece-count base shortened smoothly as |rootQ| grows): see GameLimitHorizon.

  // Per-game controller state. Committed on the first move of each game
  // (or lazily on the first call if the game start was never signaled, e.g. UCI mid-game).
  bool scheduleInitialized;
  float scheduleR0;       // remaining clock when the schedule was committed
  float scheduleM0;       // expected own moves over which to glide to the reserve
  float scheduleReserve;  // terminal clock target
  int ownMoveIndex;       // m: own moves allocated so far this game (warm calls only)


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="aggressiveness"></param>
  public ManagerGameLimitTest(float aggressiveness = 1.0f)
  {
    Aggressiveness = aggressiveness;
  }


  public ManagerGameLimitOutputs ComputeMoveAllocation(ManagerGameLimitInputs inputs, bool applyEarlySmoothing = true)
  {
    // applyEarlySmoothing == false marks a recomputation for the same move (e.g. cold-start
    // reallocation after a reused graph was abandoned): state must be neither reset nor advanced.
    if (applyEarlySmoothing && inputs.IsFirstMoveOfGame)
    {
      scheduleInitialized = false;
    }

    bool isNodes = SearchLimit.TypeIsNodesLimit(inputs.TargetLimitType);
    float minAllocation = isNodes ? 1 : MIN_TIME;

    ManagerGameLimitOutputs Return(float value, float extensionFraction, string path,
                                   float horizon = float.NaN, float targetR = float.NaN,
                                   float drainTerm = float.NaN, float errTerm = float.NaN,
                                   float winFactor = float.NaN)
    {
      float finalValue = Math.Max(minAllocation, value);
      ManagerGameLimitOutputs outputs = new(new SearchLimit(inputs.TargetLimitType, finalValue,
                                                            fractionExtensibleIfNeeded: extensionFraction,
                                                            maxTreeNodes: inputs.MaxTreeNodesSelf,
                                                            maxTreeVisits: inputs.MaxTreeVisitsSelf));
      if (CaptureDiagnostics)
      {
        outputs.DiagnosticText = BuildDiagnosticString(inputs, path, finalValue, horizon, targetR,
                                                       drainTerm, errTerm, winFactor);
      }
      return outputs;
    }

    // If this is the last move to go, use almost all available time.
    if (inputs.MaxMovesToGo.HasValue && inputs.MaxMovesToGo < 2)
    {
      return Return(inputs.RemainingFixedSelf * 0.98f, 0, "traj-last-move");
    }

    // Panic / near-exhaustion short-circuits (thresholds identical to the default manager).
    float incrementMeaningfulThreshold = isNodes ? 1 : 0.01f;
    bool hasMeaningfulIncrement = inputs.IncrementSelf > incrementMeaningfulThreshold;
    if (Panic(inputs))
    {
      return Return(inputs.RemainingFixedSelf * (hasMeaningfulIncrement ? 0.50f : 0.01f), 0.0f, "traj-panic");
    }
    else if (NearExhaustion(inputs))
    {
      return Return(inputs.RemainingFixedSelf * (hasMeaningfulIncrement ? 0.70f : 0.03f), 0.2f, "traj-near-exhaustion");
    }

    float remaining = inputs.RemainingFixedSelf;
    float horizonNow = HorizonOwnMoves(inputs);

    if (!scheduleInitialized)
    {
      scheduleR0 = remaining;
      scheduleM0 = horizonNow;
      scheduleReserve = Math.Max(RESERVE_MIN_INCREMENTS * inputs.IncrementSelf,
                                 RESERVE_FRACTION_OF_START * scheduleR0);
      ownMoveIndex = 0;
      scheduleInitialized = true;
    }

    // A cold recompute (applyEarlySmoothing == false) re-derives the allocation for the move
    // most recently allocated, so it must reuse that move's index rather than the next one.
    int m = applyEarlySmoothing ? ownMoveIndex : Math.Max(0, ownMoveIndex - 1);
    if (applyEarlySmoothing)
    {
      ownMoveIndex++;
    }

    // Committed schedule value at this move (glides to the reserve at m == M0, then holds it).
    float scheduleFraction = MathF.Max(0, 1f - m / scheduleM0);
    float targetRemaining = scheduleReserve + (scheduleR0 - scheduleReserve) * MathF.Pow(scheduleFraction, SHAPE_P);

    // Planned glide step from the LIVE horizon, plus correction toward the committed schedule.
    float drain = (remaining - scheduleReserve) * (SHAPE_P / horizonNow);
    if (m == 0)
    {
      drain *= FIRST_MOVE_FACTOR;
    }

    // Anti-windup: a surplus (ahead of schedule) is re-spent in full at ERROR_GAIN, but a
    // deficit (behind schedule) may slow spending by at most half the planned drain, so a
    // large transient error can never collapse the allocation to (near) zero.
    float correction = ERROR_GAIN * (remaining - targetRemaining);
    correction = MathF.Max(correction, -0.5f * drain);

    float winFactorValue = WinnownessFactor(inputs);
    float alloc = FRACTION_OF_INCREMENT * inputs.IncrementSelf
                + Aggressiveness * (drain + correction) * winFactorValue;

    alloc = Math.Min(alloc, remaining * MAX_FRACTION_REMAINING_PER_MOVE);
    return Return(alloc, 0.6f, "traj", horizonNow, targetRemaining,
                  drain, correction, winFactorValue);
  }


  /// <summary>
  /// Live estimate of the number of OWN moves remaining in the game
  /// (the shared NN-free estimator).
  /// </summary>
  static float HorizonOwnMoves(ManagerGameLimitInputs inputs)
    => GameLimitHorizon.EstimateOwnMovesRemaining(inputs);


  /// <summary>
  /// Winningness/courtesy shaping factor (same ladder as the default manager): spend less when
  /// clearly winning (or as a courtesy in trivially-won positions with ample clock),
  /// slightly more when losing. Applied to the controller terms only, never to the increment.
  /// </summary>
  static float WinnownessFactor(ManagerGameLimitInputs inputs)
  {
    const float COURTESY_Q_400CP = 0.8630f;
    const float COURTESY_Q_700CP = 0.9227f;
    const float COURTESY_MIN_CLOCK_SECONDS = 180f;

    bool courtesyEligible = inputs.TargetLimitType == SearchLimitType.SecondsPerMove
                         && inputs.RemainingFixedSelf - inputs.IncrementSelf >= COURTESY_MIN_CLOCK_SECONDS;

    if (courtesyEligible && inputs.RootQ >= COURTESY_Q_700CP)
    {
      return 0.60f;
    }
    else if (courtesyEligible && inputs.RootQ >= COURTESY_Q_400CP)
    {
      return 0.80f;
    }

    return inputs.RootQ switch
    {
      < -0.75f => 1.10f,
      < -0.50f => 1.05f,
      > 0.75f => 0.90f,
      > 0.50f => 0.95f,
      _ => 1.0f
    };
  }


  static bool Panic(ManagerGameLimitInputs inputs)
    => inputs.TargetLimitType == SearchLimitType.NodesPerMove
         ? inputs.RemainingFixedSelf + inputs.IncrementSelf < 50
         : (inputs.RemainingFixedSelf + inputs.IncrementSelf) < 0.25;

  static bool NearExhaustion(ManagerGameLimitInputs inputs)
    => inputs.TargetLimitType == SearchLimitType.NodesPerMove
         ? (inputs.RemainingFixedSelf + inputs.IncrementSelf) < 200
         : (inputs.RemainingFixedSelf + inputs.IncrementSelf) < 1;


  /// <summary>
  /// Builds the per-move diagnostic block recorded in the minilog LIMITS section.
  /// Uses the same "[LimitCalc]" prefix as the default manager so log tooling matches both.
  /// </summary>
  string BuildDiagnosticString(ManagerGameLimitInputs inputs, string path, float finalValue,
                               float horizon, float targetR, float drainTerm, float errTerm,
                               float winFactor)
  {
    string unit = SearchLimit.TypeIsNodesLimit(inputs.TargetLimitType) ? "nodes" : "sec";
    static string F(float v, string fmt = "F3") => float.IsNaN(v) ? "n/a" : v.ToString(fmt);

    System.Text.StringBuilder sb = new();
    sb.AppendLine($"[LimitCalc] path={path}  type={inputs.TargetLimitType}  "
                + $"=> ALLOC {F(finalValue)} {unit}  aggr={Aggressiveness:F3}");
    sb.AppendLine($"           in: rootN={inputs.RootN} rootQ={F(inputs.RootQ, "F4")} "
                + $"remFixed={F(inputs.RemainingFixedSelf)} incr={F(inputs.IncrementSelf)} "
                + $"pieces={inputs.StartPos.PieceCount} firstMove={inputs.IsFirstMoveOfGame} "
                + $"movesToGo={(inputs.MaxMovesToGo.HasValue ? inputs.MaxMovesToGo.Value.ToString() : "-")}");
    sb.Append($"           traj: m={Math.Max(0, ownMoveIndex - 1)} R0={F(scheduleR0, "F1")} M0={F(scheduleM0, "F1")} "
                + $"reserve={F(scheduleReserve, "F1")} Rstar={F(targetR, "F1")} horizon={F(horizon, "F1")} "
                + $"drain={F(drainTerm)} gainErr={F(errTerm)} winFactor={F(winFactor, "F2")}");
    return sb.ToString();
  }
}
