#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

namespace Ceres.Chess.GameEngines
{
  /// <summary>
  /// Compact summary of the root move statistics from an engine's most recently completed search,
  /// sufficient to characterize why a move was chosen (used by tournament blunder diagnostics).
  ///
  /// All Q values are stated from the perspective of the side to move at the search root
  /// (i.e. the engine's own perspective), so higher is better for the engine.
  /// </summary>
  public sealed record GameEngineRootMoveSummary
  {
    /// <summary>Visit count at the search root.</summary>
    public int RootN;

    /// <summary>Search stop status (engine specific descriptive string), or null if unavailable.</summary>
    public string StopStatus;

    /// <summary>Move actually selected, in SAN.</summary>
    public string PlayedMoveSAN;

    /// <summary>Backed up Q of the move actually selected.</summary>
    public float PlayedMoveQ;

    /// <summary>Visits to the move actually selected.</summary>
    public int PlayedMoveN;

    /// <summary>Policy prior of the move actually selected (0-1).</summary>
    public float PlayedMoveP;

    /// <summary>
    /// True if a best alternative root move (best Q among sufficiently visited moves
    /// other than the one played) was identified.
    /// </summary>
    public bool HaveBestAlternative;

    /// <summary>Best alternative root move, in SAN.</summary>
    public string BestAlternativeSAN;

    /// <summary>Backed up Q of the best alternative root move.</summary>
    public float BestAlternativeQ;

    /// <summary>Visits to the best alternative root move.</summary>
    public int BestAlternativeN;

    /// <summary>Policy prior of the best alternative root move (0-1).</summary>
    public float BestAlternativeP;

    /// <summary>
    /// True if the move played has a sufficiently visited best reply,
    /// allowing the backup dilution check below.
    /// </summary>
    public bool HaveBestReply;

    /// <summary>Opponent's best reply to the move played, in SAN.</summary>
    public string BestReplySAN;

    /// <summary>
    /// Q after the opponent's best reply to the move played (engine's own perspective).
    /// A value materially below PlayedMoveQ means the search located a refutation which the
    /// visit-weighted average at the root has not yet absorbed.
    /// </summary>
    public float BestReplyQ;

    /// <summary>Move with the highest policy prior at the root, in SAN.</summary>
    public string TopPolicyMoveSAN;

    /// <summary>Amount by which PlayedMoveQ exceeds BestAlternativeQ (negative if an alternative scored better).</summary>
    public float QGapToBestAlternative => PlayedMoveQ - BestAlternativeQ;

    /// <summary>
    /// Amount by which the backed up Q of the played move is more optimistic than the value
    /// after the opponent's best reply found by the search.
    /// </summary>
    public float BackupOptimism => PlayedMoveQ - BestReplyQ;
  }
}
