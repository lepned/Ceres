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

using Ceres.Chess.MoveGen.Test;

#endregion

namespace Ceres.Commands
{
  public record FeatureBenchmarkPerft
  {
    /// <summary>
    /// Constructor which parses arguments.
    /// </summary>
    /// <param name="args"></param>
    /// <returns></returns>
    public static void Execute(string args)
    {
      const int MAX_DEPTH = 7;

      string[] perftParts = args.Split(" ");
      if (perftParts.Length == 1 && int.TryParse(args, out int n1) && n1 > 1 && n1 < MAX_DEPTH)
      {
        MGMoveGenTest.RunChess960Verification(n1, 960);
      }
      else if (perftParts.Length == 2 && int.TryParse(perftParts[1], out int n2) && n2 > 1 && n2 < MAX_DEPTH)
      {
        MGMoveGenTest.RunChess960Verification(n2, int.Parse(perftParts[0]));
      }
      else
      {
        Console.WriteLine("No valid depth number given to Perft, defaulting to depth 5 and 960 positions");
        MGMoveGenTest.RunChess960Verification(5, 960);
      }
    }
  }
}
