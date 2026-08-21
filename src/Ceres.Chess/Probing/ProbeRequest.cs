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

using System.Threading;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;

#endregion

namespace Ceres.Chess.Probing;

/// <summary>
/// One asynchronous probe request against a snapshot of a search-graph position.
/// Immutable. The requester identifies the position both by its reconstructed board state
/// (position + history, sufficient for exact repetition/50-move handling in the prober) and
/// by an opaque correlation token it can use to re-find its own bookkeeping on completion.
/// </summary>
public sealed record ProbeRequest
{
  /// <summary>Monotonically increasing request ID (unique within the process).</summary>
  public required long RequestID { get; init; }

  /// <summary>Opaque correlation token round-tripped to the result (not interpreted by the prober).</summary>
  public required object RequesterToken { get; init; }

  /// <summary>Position to search, including prior game/search history
  /// (sufficient for repetition and 50-move handling).</summary>
  public required PositionWithHistory Position { get; init; }

  /// <summary>The move currently dominating MCGS visits at this node (the move under test),
  /// or default if the requester wants an untargeted best-move probe.</summary>
  public MGMove DominantMove { get; init; }

  /// <summary>Approximate search effort in prober nodes.</summary>
  public required int NodeBudget { get; init; }

  /// <summary>Cancellation: the requester cancels all outstanding probes at end of search.</summary>
  public required CancellationToken CancellationToken { get; init; }
}
