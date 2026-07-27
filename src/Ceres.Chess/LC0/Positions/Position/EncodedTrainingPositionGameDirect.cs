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
using System.Linq;

#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Wraps an array of EncodedTrainingPosition objects as an EncodedTrainingPositionGameBase.
  /// </summary>
  public class EncodedTrainingPositionGameDirect : EncodedTrainingPositionGame
  {
    Memory<EncodedTrainingPosition> positions;
    Memory<EncodedTrainingPositionExtraV7> extrasV7;

    public EncodedTrainingPositionGameDirect(Memory<EncodedTrainingPosition> positions,
                                             Memory<EncodedTrainingPositionExtraV7> extrasV7 = default)
    {
      this.positions = positions;
      this.extrasV7 = extrasV7;
      if (positions.Length == 0)
      {
        throw new ArgumentException(nameof(positions), "length zero");
      }
      if (!extrasV7.IsEmpty && extrasV7.Length != positions.Length)
      {
        throw new ArgumentException(nameof(extrasV7), "length mismatch with positions");
      }
    }

    public override int NumPositions => positions.Length;

    public override int Version => positions.Span[0].Version;

    public override int InputFormat => positions.Span[0].InputFormat;

    public override EncodedPolicyVector PolicyAtIndex(int index) => positions.Span[index].Policies;

    protected override ref readonly EncodedPositionWithHistory PositionRawMirroredRefAtIndex(int index) => ref positions.Span[index].PositionWithBoards;

    public override bool HasExtraV7 => !extrasV7.IsEmpty;

    public override EncodedTrainingPositionExtraV7 ExtraV7AtIndex(int index) => extrasV7.Span[index];

    public override string ToString()
    {
      double avgNumVisits = positions.Span.ToArray().Average(p => p.PositionWithBoards.MiscInfo.InfoTraining.NumVisits);
      return $"<EncodedPositionWithHistory {NumPositions} moves, {Math.Round(avgNumVisits, 0)} average visits>";
    }
  }
}
