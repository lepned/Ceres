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

using System.Runtime.InteropServices;

#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Origin of the result_q/result_d/plies_left values in a V7 training record
  /// (value of ZProvenance, produced by the V7 rescorer).
  /// </summary>
  public enum EncodedTrainingPositionV7ZProvenance : byte
  {
    OriginalResult = 0,       // unmodified game outcome
    TablebaseRescore = 1,     // syzygy WDL/DTZ overwrite (takes precedence at that record)
    DeblunderNoise = 2,       // deblunder z-propagation, noise trigger
    DeblunderUnintended = 3,  // deblunder z-propagation, unintended cross-ply trigger
    Op1EightManRelabel = 4,   // 8-man op1 relabel (op1wins policy)
  }


  /// <summary>
  /// The 40-byte tail appended to each V6 record in V7 training data
  /// (e.g. produced by lc0-rescorer-v7), holding extra rescorer-computed fields.
  /// Byte layout matches the file format exactly (offsets +8356..+8395 of a record).
  /// All q-like values are relative to the side to move of their record.
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  public readonly struct EncodedTrainingPositionExtraV7
  {
    public const int SIZE = 40;

    /// <summary>
    /// Short-term draw probability: backward EMA (alpha = 5/6 on the carry) of root_d,
    /// no sign alternation, clamped >= 0. [+8356]
    /// </summary>
    public readonly float DShortTerm;

    /// <summary>
    /// Move index (lc0 policy-index encoding) actually played in the NEXT record
    /// (the opponent's reply); 0xFFFF if none (last record). [+8360]
    /// </summary>
    public readonly ushort OppPlayedIndex;

    /// <summary>
    /// Move played two records ahead (side-to-move's own next move); 0xFFFF if none. [+8362]
    /// </summary>
    public readonly ushort NextPlayedIndex;

    /// <summary>
    /// z-provenance of the record's result_* fields (see EncodedTrainingPositionV7ZProvenance),
    /// stored as a float in the file. [+8364]
    /// </summary>
    public readonly float ZProvenanceRaw;

    /// <summary>
    /// Censored short-term value: backward EMA of root_q (alpha = 5/6 on the carry,
    /// sign-alternating, STM-relative) whose carry re-initializes to best_q at every
    /// record where a deblunder trigger fired — never blends across a detected blunder. [+8368]
    /// </summary>
    public readonly float CensoredQShortTerm;

    /// <summary>
    /// Censored short-term draw probability (same censored walk over root_d/best_d, clamped >= 0). [+8372]
    /// </summary>
    public readonly float CensoredDShortTerm;

    /// <summary>
    /// Evaluation after the played move from the mover's perspective:
    /// -(next record's best_q); NaN at the last record. [+8376]
    /// </summary>
    public readonly float QAfterPlayedMove;

    public readonly float Reserved4;
    public readonly float Reserved5;
    public readonly float Reserved6;
    public readonly float Reserved7;

    /// <summary>
    /// ZProvenance as the enum (file stores small nonnegative integers as float).
    /// </summary>
    public EncodedTrainingPositionV7ZProvenance ZProvenance => (EncodedTrainingPositionV7ZProvenance)(byte)ZProvenanceRaw;

    /// <summary>
    /// Sentinel value of OppPlayedIndex/NextPlayedIndex indicating no such move exists.
    /// </summary>
    public const ushort NO_MOVE_INDEX = 0xFFFF;
  }
}
