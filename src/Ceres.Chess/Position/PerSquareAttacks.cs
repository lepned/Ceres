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
    /// Flip files of a bitboard: a1↔h1, b1↔g1, ..., a8↔h8.
    /// Equivalent to bit-XOR-7 on every set bit, done in parallel via the
    /// standard "delta swap" / file-mirror bit-twiddle.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static BitBoard FlipFiles(BitBoard bb)
    {
      return ((bb & 0x0101010101010101UL) << 7)
           | ((bb & 0x0202020202020202UL) << 5)
           | ((bb & 0x0404040404040404UL) << 3)
           | ((bb & 0x0808080808080808UL) << 1)
           | ((bb & 0x1010101010101010UL) >> 1)
           | ((bb & 0x2020202020202020UL) >> 3)
           | ((bb & 0x4040404040404040UL) >> 5)
           | ((bb & 0x8080808080808080UL) >> 7);
    }

    /// <summary>
    /// For the given position, count for every square how many WHITE pieces and how many
    /// BLACK pieces attack it. Writes into the two 64-element spans (must be pre-zeroed
    /// by caller, OR overwritten — this method zero-fills before writing).
    ///
    /// Convention: WHITE/BLACK here are the REAL board colors, not us-to-move
    /// canonical orientation. Caller is responsible for swapping if a different
    /// orientation is needed.
    ///
    /// Output square indexing is python-chess / TPG-standard convention:
    ///   a1 = sq 0, h1 = sq 7, a8 = sq 56, h8 = sq 63
    /// Note that Ceres internal MGPosition bitboards use a FILE-REVERSED bit layout
    /// (a1 = bit 7, h1 = bit 0); we file-flip on entry so all internal logic
    /// (attack tables, ray scans) can use standard convention.
    /// </summary>
    public static void Compute(in MGPosition pos,
                                Span<byte> whiteAttackerCount,  // length 64
                                Span<byte> blackAttackerCount)  // length 64
    {
      // File-flip on entry: Ceres bitboards have file 0 at bit 7, file 7 at bit 0.
      BitBoard A = FlipFiles(pos.A);
      BitBoard B = FlipFiles(pos.B);
      BitBoard C = FlipFiles(pos.C);
      BitBoard D = FlipFiles(pos.D);

      // Per-color piece-type bitboards (excludes EP-marker squares: EP has A=B=1, C=0):
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

      ComputeFromBitboards(wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK,
                           whiteAttackerCount, blackAttackerCount);
    }

    /// <summary>
    /// Compute attacker counts directly from 12 per-piece-type bitboards. Bitboards must
    /// be in STANDARD convention (a1=bit 0, h8=bit 63) — i.e. python-chess / TPG layout.
    /// Used by the V2→V3 TPG upgrade path, which extracts bitboards from the one-hot
    /// piece bytes already stored in V2 records.
    /// </summary>
    public static void ComputeFromBitboards(
        BitBoard wP, BitBoard wN, BitBoard wB, BitBoard wR, BitBoard wQ, BitBoard wK,
        BitBoard bP, BitBoard bN, BitBoard bB, BitBoard bR, BitBoard bQ, BitBoard bK,
        Span<byte> whiteAttackerCount, Span<byte> blackAttackerCount)
    {
      if (whiteAttackerCount.Length != 64) throw new ArgumentException("whiteAttackerCount must be length 64");
      if (blackAttackerCount.Length != 64) throw new ArgumentException("blackAttackerCount must be length 64");

      whiteAttackerCount.Clear();
      blackAttackerCount.Clear();

      BitBoard occupancy = wP | wN | wB | wR | wQ | wK | bP | bN | bB | bR | bQ | bK;

      // Walk each piece, accumulate per-target-square counters.
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

    /// <summary>
    /// V2→V3 TPG upgrade helper. Reads the 64×137-byte square block from a V2 TPG record,
    /// extracts the per-square piece one-hot (bytes [0:13]: 0=empty, 1-6=our P,N,B,R,Q,K,
    /// 7-12=opp P,N,B,R,Q,K), builds per-piece bitboards, and computes "our"/"opp" attacker
    /// counts per TPG square.
    ///
    /// Since V2 records are already us-to-move oriented (our pieces at low TPG ranks), the
    /// "white" attackers in the bitboard sense ARE our attackers in TPG sense — no
    /// orientation flip needed in the caller. Output spans directly correspond to
    /// per-TPG-slot our-attackers / opp-attackers.
    /// </summary>
    public static void ComputeFromTpgSquareBytes(ReadOnlySpan<byte> squareBytes,  // 64 * 137 contiguous
                                                 Span<byte> ourAttackerCount,     // length 64
                                                 Span<byte> oppAttackerCount,     // length 64)
                                                 int bytesPerSquare = 137)
    {
      if (squareBytes.Length < 64 * bytesPerSquare)
      {
        throw new ArgumentException($"squareBytes too short ({squareBytes.Length}, need {64 * bytesPerSquare})");
      }

      // Extract per-piece-type bitboards from the V2 one-hot.
      // V2 byte encoding (per ByteScaled convention): byte=100 means "this class present" (float 1.0
      // after /100). We use >50 as the threshold — any byte ≥50 means "set".
      BitBoard ourP = 0, ourN = 0, ourB = 0, ourR = 0, ourQ = 0, ourK = 0;
      BitBoard oppP = 0, oppN = 0, oppB = 0, oppR = 0, oppQ = 0, oppK = 0;
      for (int sq = 0; sq < 64; sq++)
      {
        int off = sq * bytesPerSquare;
        BitBoard bit = 1UL << sq;
        // Slots: 0=empty, 1=ourP, 2=ourN, 3=ourB, 4=ourR, 5=ourQ, 6=ourK,
        //        7=oppP, 8=oppN, 9=oppB, 10=oppR, 11=oppQ, 12=oppK
        if      (squareBytes[off + 1]  > 50) ourP |= bit;
        else if (squareBytes[off + 2]  > 50) ourN |= bit;
        else if (squareBytes[off + 3]  > 50) ourB |= bit;
        else if (squareBytes[off + 4]  > 50) ourR |= bit;
        else if (squareBytes[off + 5]  > 50) ourQ |= bit;
        else if (squareBytes[off + 6]  > 50) ourK |= bit;
        else if (squareBytes[off + 7]  > 50) oppP |= bit;
        else if (squareBytes[off + 8]  > 50) oppN |= bit;
        else if (squareBytes[off + 9]  > 50) oppB |= bit;
        else if (squareBytes[off + 10] > 50) oppR |= bit;
        else if (squareBytes[off + 11] > 50) oppQ |= bit;
        else if (squareBytes[off + 12] > 50) oppK |= bit;
        // empty or EP marker — no piece bitboard updated
      }

      // Compute attacker counts. "Our" bitboards take the WHITE-pawn-attack pattern (which
      // attacks forward-up), "opp" take the BLACK pawn pattern — since the canonical board
      // has our pieces on low ranks moving up.
      ComputeFromBitboards(ourP, ourN, ourB, ourR, ourQ, ourK,
                           oppP, oppN, oppB, oppR, oppQ, oppK,
                           ourAttackerCount, oppAttackerCount);
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
