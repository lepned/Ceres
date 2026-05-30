#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
*/

#endregion

#region Using directives

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using Ceres.Chess.MoveGen;

using BitBoard = System.UInt64;

#endregion

namespace Ceres.Chess.PositionDataInfo
{
  /// <summary>
  /// Computes per-square attacker counts (one byte per square per color) for use as
  /// augmented input features in CeresTrain transformer networks.
  ///
  /// The output for each square is the popcount of pieces of a given color whose
  /// attack pattern includes that square — matches python-chess
  /// <c>chess.Board.attackers_mask(color, square).popcount()</c>.
  /// Includes "defenders" (pieces of color X attacking own pieces on the square).
  ///
  /// Square indexing follows Ceres / python-chess convention: a1 = 0, h8 = 63.
  ///
  /// The bitboard primitives (knight/king/pawn pre-computed tables; sliding-piece
  /// iterative ray scan with blocker handling) are inlined here for clarity and
  /// independent validatability against python-chess (see equality test). Magic
  /// bitboards would be faster but harder to verify; this implementation is fast
  /// enough for inference-time use (microseconds per position).
  /// </summary>
  public static class PerSquareAttacks
  {
    // ---------- Static pre-computed attack tables ------------------------------

    /// <summary>For each square s in [0, 64): bitmask of squares a knight on s attacks.</summary>
    private static readonly BitBoard[] KNIGHT_ATTACKS = InitKnightAttacks();

    /// <summary>For each square s: bitmask of squares a king on s attacks.</summary>
    private static readonly BitBoard[] KING_ATTACKS = InitKingAttacks();

    /// <summary>For each square s: bitmask of squares a WHITE pawn on s attacks.</summary>
    private static readonly BitBoard[] PAWN_ATTACKS_WHITE = InitPawnAttacks(isWhite: true);

    /// <summary>For each square s: bitmask of squares a BLACK pawn on s attacks.</summary>
    private static readonly BitBoard[] PAWN_ATTACKS_BLACK = InitPawnAttacks(isWhite: false);

    // Sliding-piece directions (file delta, rank delta).
    private static readonly (int df, int dr)[] ROOK_DIRS   = new[] { (0, 1), (0, -1), (1, 0), (-1, 0) };
    private static readonly (int df, int dr)[] BISHOP_DIRS = new[] { (1, 1), (1, -1), (-1, 1), (-1, -1) };

    private static BitBoard[] InitKnightAttacks()
    {
      var t = new BitBoard[64];
      var deltas = new (int df, int dr)[] { (1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1) };
      for (int sq = 0; sq < 64; sq++)
      {
        int f = sq & 7, r = sq >> 3;
        BitBoard m = 0UL;
        foreach (var (df, dr) in deltas)
        {
          int nf = f + df, nr = r + dr;
          if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) m |= 1UL << (nr * 8 + nf);
        }
        t[sq] = m;
      }
      return t;
    }

    private static BitBoard[] InitKingAttacks()
    {
      var t = new BitBoard[64];
      for (int sq = 0; sq < 64; sq++)
      {
        int f = sq & 7, r = sq >> 3;
        BitBoard m = 0UL;
        for (int df = -1; df <= 1; df++)
          for (int dr = -1; dr <= 1; dr++)
          {
            if (df == 0 && dr == 0) continue;
            int nf = f + df, nr = r + dr;
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) m |= 1UL << (nr * 8 + nf);
          }
        t[sq] = m;
      }
      return t;
    }

    private static BitBoard[] InitPawnAttacks(bool isWhite)
    {
      var t = new BitBoard[64];
      int dr = isWhite ? 1 : -1;   // white pawns attack forward (rank +1), black attacks rank -1
      for (int sq = 0; sq < 64; sq++)
      {
        int f = sq & 7, r = sq >> 3;
        BitBoard m = 0UL;
        foreach (int df in new[] { -1, 1 })
        {
          int nf = f + df, nr = r + dr;
          if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) m |= 1UL << (nr * 8 + nf);
        }
        t[sq] = m;
      }
      return t;
    }

    /// <summary>
    /// Compute the attack mask of a sliding piece (rook/bishop/queen) on a given
    /// source square, given the full board occupancy. The attack mask includes
    /// the first blocker in each direction (where the slider could capture).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static BitBoard SlidingAttacks(int sourceSq, BitBoard occupancy, (int df, int dr)[] dirs)
    {
      BitBoard attacks = 0UL;
      int sf = sourceSq & 7, sr = sourceSq >> 3;
      foreach (var (df, dr) in dirs)
      {
        int nf = sf + df, nr = sr + dr;
        while (nf >= 0 && nf < 8 && nr >= 0 && nr < 8)
        {
          int nsq = nr * 8 + nf;
          attacks |= 1UL << nsq;
          if ((occupancy & (1UL << nsq)) != 0) break;   // blocker — include it, stop ray
          nf += df; nr += dr;
        }
      }
      return attacks;
    }

    // ---------- Main entry point ------------------------------------------------

    /// <summary>
    /// For the given position, count for every square how many WHITE pieces and how many
    /// BLACK pieces attack it. Writes into the two 64-element spans (must be pre-zeroed
    /// by caller, OR overwritten — this method zero-fills before writing).
    ///
    /// Convention: WHITE/BLACK here are the REAL board colors, not us-to-move
    /// canonical orientation. Caller is responsible for swapping if a different
    /// orientation is needed.
    /// </summary>
    public static void Compute(in MGPosition pos,
                                Span<byte> whiteAttackerCount,  // length 64
                                Span<byte> blackAttackerCount)  // length 64
    {
      if (whiteAttackerCount.Length != 64) throw new ArgumentException("whiteAttackerCount must be length 64");
      if (blackAttackerCount.Length != 64) throw new ArgumentException("blackAttackerCount must be length 64");

      whiteAttackerCount.Clear();
      blackAttackerCount.Clear();

      // Bitplane representation per MGPosition (see MGPosition.cs):
      //   D (msb) = Color (1=Black, 0=White)
      //   C       = 1=Straight-moving (Q or R)
      //   B       = 1=Diagonal-moving (Q or B)
      //   A (lsb) = 1=Pawn
      // Piece codes:
      //   1 (0001) = WP, 2 (0010) = WB, 3 (0011) = W EP, 4 (0100) = WR,
      //   5 (0101) = WN, 6 (0110) = WQ, 7 (0111) = WK,
      //   9..15 = same with D=1 for black.
      //   En-passant square codes (3, 11) are markers, not pieces.

      BitBoard A = pos.A, B = pos.B, C = pos.C, D = pos.D;

      // Per-color piece-type bitboards (excludes EP-marker squares since EP has A=B=1, C=0):
      BitBoard wP = A & ~B & ~C & ~D;     // White Pawn (code 1)
      BitBoard wB = ~A & B & ~C & ~D;     // White Bishop (code 2)
      BitBoard wR = ~A & ~B & C & ~D;     // White Rook (code 4)
      BitBoard wN = A & ~B & C & ~D;      // White Knight (code 5)
      BitBoard wQ = ~A & B & C & ~D;      // White Queen (code 6)
      BitBoard wK = A & B & C & ~D;       // White King (code 7)
      BitBoard bP = A & ~B & ~C & D;      // Black Pawn (code 9)
      BitBoard bB = ~A & B & ~C & D;      // Black Bishop (code 10)
      BitBoard bR = ~A & ~B & C & D;      // Black Rook (code 12)
      BitBoard bN = A & ~B & C & D;       // Black Knight (code 13)
      BitBoard bQ = ~A & B & C & D;       // Black Queen (code 14)
      BitBoard bK = A & B & C & D;        // Black King (code 15)

      // True occupancy (excludes EP markers — those have A=1,B=1,C=0 patterns).
      BitBoard whitePieces = wP | wB | wR | wN | wQ | wK;
      BitBoard blackPieces = bP | bB | bR | bN | bQ | bK;
      BitBoard occupancy = whitePieces | blackPieces;

      // Walk each piece, OR its attack-mask into a per-color "all attacks" map by
      // incrementing per-target-square counters. We need COUNTS, not just unions —
      // because two pieces of the same color could attack the same square.

      AddAttacks(whiteAttackerCount, wP, PAWN_ATTACKS_WHITE);
      AddAttacks(whiteAttackerCount, wN, KNIGHT_ATTACKS);
      AddAttacks(whiteAttackerCount, wK, KING_ATTACKS);
      AddSlidingAttacks(whiteAttackerCount, wB, occupancy, BISHOP_DIRS);
      AddSlidingAttacks(whiteAttackerCount, wR, occupancy, ROOK_DIRS);
      AddSlidingAttacks(whiteAttackerCount, wQ, occupancy, BISHOP_DIRS);
      AddSlidingAttacks(whiteAttackerCount, wQ, occupancy, ROOK_DIRS);

      AddAttacks(blackAttackerCount, bP, PAWN_ATTACKS_BLACK);
      AddAttacks(blackAttackerCount, bN, KNIGHT_ATTACKS);
      AddAttacks(blackAttackerCount, bK, KING_ATTACKS);
      AddSlidingAttacks(blackAttackerCount, bB, occupancy, BISHOP_DIRS);
      AddSlidingAttacks(blackAttackerCount, bR, occupancy, ROOK_DIRS);
      AddSlidingAttacks(blackAttackerCount, bQ, occupancy, BISHOP_DIRS);
      AddSlidingAttacks(blackAttackerCount, bQ, occupancy, ROOK_DIRS);
    }

    // For non-sliders: iterate pieces, look up attack mask from precomputed table,
    // increment counter[target_sq] for each target bit.
    private static void AddAttacks(Span<byte> counter, BitBoard piecesOfType, BitBoard[] attackTable)
    {
      while (piecesOfType != 0)
      {
        int sq = BitOperations.TrailingZeroCount(piecesOfType);
        piecesOfType &= piecesOfType - 1;
        BitBoard targets = attackTable[sq];
        while (targets != 0)
        {
          int t = BitOperations.TrailingZeroCount(targets);
          targets &= targets - 1;
          counter[t]++;
        }
      }
    }

    private static void AddSlidingAttacks(Span<byte> counter, BitBoard piecesOfType,
                                          BitBoard occupancy, (int df, int dr)[] dirs)
    {
      while (piecesOfType != 0)
      {
        int sq = BitOperations.TrailingZeroCount(piecesOfType);
        piecesOfType &= piecesOfType - 1;
        BitBoard targets = SlidingAttacks(sq, occupancy, dirs);
        while (targets != 0)
        {
          int t = BitOperations.TrailingZeroCount(targets);
          targets &= targets - 1;
          counter[t]++;
        }
      }
    }
  }
}
