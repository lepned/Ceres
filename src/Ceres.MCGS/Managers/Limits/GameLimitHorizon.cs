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

namespace Ceres.MCGS.Managers.Limits;

/// <summary>
/// Shared, deliberately NN-free estimator of the number of OWN moves remaining in a game
/// (no reliance on a moves-left head, which is often absent and of poor quality):
/// a piece-count base, shortened smoothly as |rootQ| grows since decisive positions end
/// games early (adjudication or collapses). Used by the limits managers both to plan
/// spending horizons and to gauge game progress.
/// </summary>
public static class GameLimitHorizon
{
  const float BASE_OWN_MOVES = 14f;
  const float OWN_MOVES_PER_PIECE = 1.6f;
  const float Q_SQUARED_SHORTENING = 1.5f;
  const float MIN_OWN_MOVES = 6f;
  const float MAX_OWN_MOVES = 80f;


  /// <summary>
  /// Returns the estimated number of own moves remaining, bounded, and capped by an
  /// explicit movestogo when one is in force.
  /// </summary>
  public static float EstimateOwnMovesRemaining(ManagerGameLimitInputs inputs)
  {
    float horizon = BASE_OWN_MOVES + OWN_MOVES_PER_PIECE * inputs.StartPos.PieceCount;

    float rootQ = inputs.RootQ;
    if (!float.IsNaN(rootQ))
    {
      horizon /= 1f + Q_SQUARED_SHORTENING * rootQ * rootQ;
    }

    if (inputs.MaxMovesToGo.HasValue)
    {
      horizon = Math.Min(horizon, Math.Max(1, inputs.MaxMovesToGo.Value - 1));
    }

    return Math.Clamp(horizon, MIN_OWN_MOVES, MAX_OWN_MOVES);
  }
}
