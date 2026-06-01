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
  /// auxiliary input features in CeresTrain transformer networks.
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

    // ============================================================================
    // V3 aux features (mobility, defender count, is-pinned, is-threatened)
    // ============================================================================

    /// <summary>
    /// One-shot computation of all V3 per-square aux features in a single pass over
    /// the position. Cheaper than 4 separate Compute*() calls because the piece-bitboard
    /// extraction + per-color attacker bitboards are shared.
    ///
    /// All 6 output spans must be length 64 (overwritten).
    /// Output convention: WHITE/BLACK are the REAL board colors. Caller maps to our/opp by side-to-move.
    ///
    /// Mobility encoding: scaled count * 100 / 27 (max plausible mobility, fits in byte 0-100).
    /// Defender encoding: count * 100 / 8 (same as our_attackers).
    /// Is-pinned encoding: 0 or 100 (boolean).
    /// Is-threatened encoding: 0 or 100 (boolean, value-aware — attacked by an opp piece of
    ///   strictly LOWER piece value than the piece on the square).
    /// </summary>
    public static void ComputeExtendedFeatures(in MGPosition pos,
                                               Span<byte> whiteAttackers,    // 64 (computed internally; used to derive defender_count)
                                               Span<byte> blackAttackers,    // 64 (computed internally; used to derive defender_count)
                                               Span<byte> mobility,          // 64 — by piece on square
                                               Span<byte> defenderCount,     // 64 — friendly attackers of piece on sq
                                               Span<byte> isPinned,          // 64 — boolean, piece is pinned to its king
                                               Span<byte> isThreatened)      // 64 — boolean, value-aware
    {
      // ---- Shared piece-bitboard extraction (file-flip on entry, same as Compute) ----
      BitBoard A = FlipFiles(pos.A);
      BitBoard B = FlipFiles(pos.B);
      BitBoard C = FlipFiles(pos.C);
      BitBoard D = FlipFiles(pos.D);

      BitBoard wP = A & ~B & ~C & ~D;
      BitBoard wB = ~A & B & ~C & ~D;
      BitBoard wR = ~A & ~B & C & ~D;
      BitBoard wN = A & ~B & C & ~D;
      BitBoard wQ = ~A & B & C & ~D;
      BitBoard wK = A & B & C & ~D;
      BitBoard bP = A & ~B & ~C & D;
      BitBoard bB = ~A & B & ~C & D;
      BitBoard bR = ~A & ~B & C & D;
      BitBoard bN = A & ~B & C & D;
      BitBoard bQ = ~A & B & C & D;
      BitBoard bK = A & B & C & D;

      ComputeExtendedFromBitboards(wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK,
                                    whiteAttackers, blackAttackers,
                                    mobility, defenderCount, isPinned, isThreatened);
    }

    /// <summary>
    /// V2→V3 TPG upgrade helper. Companion to <see cref="ComputeFromTpgSquareBytes"/>:
    /// reads 64×137-byte V2 square block + computes the 4 V3-min extended-feature spans by
    /// treating "our" pieces as white and "opp" as black (extended features are color-
    /// symmetric: a pinned piece is pinned regardless of color label; mobility/defender/
    /// threatened likewise).
    ///
    /// All output spans must be length 64. The attacker-count spans are written internally
    /// (defender_count needs them) but are no longer part of the V3 output format.
    /// </summary>
    public static void ComputeExtendedFromTpgSquareBytes(ReadOnlySpan<byte> squareBytes,
                                                          Span<byte> ourAttackerCount,
                                                          Span<byte> oppAttackerCount,
                                                          Span<byte> mobility,
                                                          Span<byte> defenderCount,
                                                          Span<byte> isPinned,
                                                          Span<byte> isThreatened,
                                                          int bytesPerSquare = 137)
    {
      if (squareBytes.Length < 64 * bytesPerSquare)
      {
        throw new ArgumentException($"squareBytes too short ({squareBytes.Length}, need {64 * bytesPerSquare})");
      }

      // Decode 12 piece bitboards from V2 one-hot (same logic as ComputeFromTpgSquareBytes).
      BitBoard ourP = 0, ourN = 0, ourB = 0, ourR = 0, ourQ = 0, ourK = 0;
      BitBoard oppP = 0, oppN = 0, oppB = 0, oppR = 0, oppQ = 0, oppK = 0;
      for (int sq = 0; sq < 64; sq++)
      {
        int off = sq * bytesPerSquare;
        BitBoard bit = 1UL << sq;
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
      }

      // "our" plays the white role for the extended-features algorithm. All 4 extra channels
      // are color-symmetric (pin = friendly king alignment; mobility/defender = own-color
      // exclusion; threat = piece-value comparison) so the role assignment is unobservable.
      ComputeExtendedFromBitboards(ourP, ourN, ourB, ourR, ourQ, ourK,
                                    oppP, oppN, oppB, oppR, oppQ, oppK,
                                    ourAttackerCount, oppAttackerCount,
                                    mobility, defenderCount, isPinned, isThreatened);
    }

    /// <summary>
    /// Shared core: computes all 6 extended-feature spans from 12 per-piece-type bitboards.
    /// Called by both <see cref="ComputeExtendedFeatures"/> (MGPosition front-end) and
    /// <see cref="ComputeExtendedFromTpgSquareBytes"/> (V2 upgrade front-end).
    /// </summary>
    private static void ComputeExtendedFromBitboards(BitBoard wP, BitBoard wN, BitBoard wB, BitBoard wR, BitBoard wQ, BitBoard wK,
                                                      BitBoard bP, BitBoard bN, BitBoard bB, BitBoard bR, BitBoard bQ, BitBoard bK,
                                                      Span<byte> whiteAttackers,
                                                      Span<byte> blackAttackers,
                                                      Span<byte> mobility,
                                                      Span<byte> defenderCount,
                                                      Span<byte> isPinned,
                                                      Span<byte> isThreatened)
    {
      BitBoard whitePieces = wP | wN | wB | wR | wQ | wK;
      BitBoard blackPieces = bP | bN | bB | bR | bQ | bK;
      BitBoard occupancy = whitePieces | blackPieces;

      // ---- 1. Attacker counts (same as Compute) ----
      ComputeFromBitboards(wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK,
                           whiteAttackers, blackAttackers);

      // ---- 2. Mobility per piece ----
      // For each piece on the board, count the squares it can MOVE to
      // (attack mask AND NOT own pieces).
      mobility.Clear();
      AddMobility(mobility, wP, ~whitePieces, PAWN_ATTACKS_WHITE);
      AddMobility(mobility, wN, ~whitePieces, KNIGHT_ATTACKS);
      AddMobility(mobility, wK, ~whitePieces, KING_ATTACKS);
      AddMobilitySlider(mobility, wB, occupancy, ~whitePieces, BISHOP_DIRS);
      AddMobilitySlider(mobility, wR, occupancy, ~whitePieces, ROOK_DIRS);
      AddMobilitySlider(mobility, wQ, occupancy, ~whitePieces, BISHOP_DIRS);
      AddMobilitySlider(mobility, wQ, occupancy, ~whitePieces, ROOK_DIRS);
      AddMobility(mobility, bP, ~blackPieces, PAWN_ATTACKS_BLACK);
      AddMobility(mobility, bN, ~blackPieces, KNIGHT_ATTACKS);
      AddMobility(mobility, bK, ~blackPieces, KING_ATTACKS);
      AddMobilitySlider(mobility, bB, occupancy, ~blackPieces, BISHOP_DIRS);
      AddMobilitySlider(mobility, bR, occupancy, ~blackPieces, ROOK_DIRS);
      AddMobilitySlider(mobility, bQ, occupancy, ~blackPieces, BISHOP_DIRS);
      AddMobilitySlider(mobility, bQ, occupancy, ~blackPieces, ROOK_DIRS);
      // Scale: raw 0..27 → byte 0..100. Use *100/27 with min(100) cap.
      for (int s = 0; s < 64; s++)
      {
        int raw = mobility[s];
        int scaled = raw * 100 / 27;
        if (scaled > 100) scaled = 100;
        mobility[s] = (byte)scaled;
      }

      // ---- 3. Defender count per occupied square ----
      // For each square with a piece, count of FRIENDLY pieces attacking that square.
      // - Our piece on sq → defenders = whiteAttackers[sq] (if our=white) or blackAttackers (if our=black)
      // - Opp piece on sq → defenders for opp = the other color's attackers
      // Empty squares: 0.
      defenderCount.Clear();
      for (int s = 0; s < 64; s++)
      {
        BitBoard bit = 1UL << s;
        int defenders;
        if ((whitePieces & bit) != 0)      defenders = whiteAttackers[s];   // white piece, white defends
        else if ((blackPieces & bit) != 0) defenders = blackAttackers[s];   // black piece, black defends
        else                                defenders = 0;
        defenderCount[s] = (byte)(defenders * 100 / 8);
      }

      // ---- 4. Is-pinned ----
      // A piece is pinned if: a friendly king lies on the same line as the piece, and
      // beyond the piece (away from king) lies an opp slider whose attack pattern matches
      // the line direction. Compute per king, scan 8 rays.
      isPinned.Clear();
      MarkPinnedAlongRays(isPinned, wK, whitePieces, blackPieces, bR | bQ, bB | bQ, occupancy);
      MarkPinnedAlongRays(isPinned, bK, blackPieces, whitePieces, wR | wQ, wB | wQ, occupancy);

      // ---- 5. Is-threatened (value-aware, NNUE-spirit) ----
      // A piece is threatened iff attacked by an opp piece of STRICTLY LOWER piece value
      // (pawn=1, knight=3, bishop=3, rook=5, queen=9, king=∞).
      // For symmetry, "king is attacked at all" → threatened (king never has lower-value attacker).
      // Pawn never strictly threatened (only pawns can attack pawns, equal trade).
      // Compute per-opp-piece-type attack bitboards, then per-square decide.
      isThreatened.Clear();
      ComputeIsThreatenedForColor(isThreatened, wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK, occupancy, attackerIsWhite: false);
      ComputeIsThreatenedForColor(isThreatened, bP, bN, bB, bR, bQ, bK, wP, wN, wB, wR, wQ, wK, occupancy, attackerIsWhite: true);
    }

    // Helper: accumulate non-slider mobility (pieces of one type vs allowed destinations)
    private static void AddMobility(Span<byte> counter, BitBoard pieces, BitBoard notOwn, BitBoard[] attackTable)
    {
      while (pieces != 0)
      {
        int sq = BitOperations.TrailingZeroCount(pieces);
        pieces &= pieces - 1;
        BitBoard targets = attackTable[sq] & notOwn;
        counter[sq] = (byte)(BitOperations.PopCount(targets));
      }
    }

    private static void AddMobilitySlider(Span<byte> counter, BitBoard pieces, BitBoard occupancy,
                                          BitBoard notOwn, (int df, int dr)[] dirs)
    {
      while (pieces != 0)
      {
        int sq = BitOperations.TrailingZeroCount(pieces);
        pieces &= pieces - 1;
        BitBoard targets = SlidingAttacks(sq, occupancy, dirs) & notOwn;
        // counter[sq] may have been set by other-direction pass for queen — accumulate
        counter[sq] = (byte)(counter[sq] + BitOperations.PopCount(targets));
      }
    }

    // Helper: mark all pieces of `friendlyPieces` that are pinned to their king by an opp slider.
    // Pin = friendly piece between king and an opp R/Q (rook-like) or B/Q (bishop-like) along same ray.
    private static void MarkPinnedAlongRays(Span<byte> pinned, BitBoard king,
                                            BitBoard friendlyPieces, BitBoard enemyPieces,
                                            BitBoard enemyRookLike, BitBoard enemyBishopLike,
                                            BitBoard occupancy)
    {
      if (king == 0) return;
      int kingSq = BitOperations.TrailingZeroCount(king);
      // 4 rook directions
      foreach (var dir in ROOK_DIRS)
      {
        TryMarkPinAlongRay(pinned, kingSq, dir, friendlyPieces, enemyRookLike, occupancy);
      }
      // 4 bishop directions
      foreach (var dir in BISHOP_DIRS)
      {
        TryMarkPinAlongRay(pinned, kingSq, dir, friendlyPieces, enemyBishopLike, occupancy);
      }
    }

    private static void TryMarkPinAlongRay(Span<byte> pinned, int kingSq, (int df, int dr) dir,
                                           BitBoard friendlyPieces, BitBoard enemySlidersOfLineType, BitBoard occupancy)
    {
      int sf = kingSq & 7, sr = kingSq >> 3;
      int nf = sf + dir.df, nr = sr + dir.dr;
      int firstFriendly = -1;
      while (nf >= 0 && nf < 8 && nr >= 0 && nr < 8)
      {
        int sq = nr * 8 + nf;
        BitBoard bit = 1UL << sq;
        if ((occupancy & bit) != 0)
        {
          if (firstFriendly < 0)
          {
            // First obstacle on the ray
            if ((friendlyPieces & bit) != 0)
            {
              firstFriendly = sq;
              // Keep scanning to find what's beyond
            }
            else
            {
              return;  // First obstacle is enemy — no pin on this ray
            }
          }
          else
          {
            // Second obstacle — is it an opp slider matching this line direction?
            if ((enemySlidersOfLineType & bit) != 0)
            {
              pinned[firstFriendly] = 100;
            }
            return;
          }
        }
        nf += dir.df; nr += dir.dr;
      }
    }

    // Compute is-threatened for one color's pieces (attackerIsWhite indicates whose attackers we check).
    // For each piece P of the defending color, check if any opp piece of strictly lower value attacks P's square.
    private static void ComputeIsThreatenedForColor(Span<byte> threatened,
                                                    BitBoard defP, BitBoard defN, BitBoard defB, BitBoard defR, BitBoard defQ, BitBoard defK,
                                                    BitBoard attP, BitBoard attN, BitBoard attB, BitBoard attR, BitBoard attQ, BitBoard attK,
                                                    BitBoard occupancy, bool attackerIsWhite)
    {
      // Compute per-attacker-piece-type attack bitboards (squares any attacker of that type covers).
      // Pawn attacks depend on color.
      BitBoard atkByP = ComputeNonSliderAttackUnion(attP, attackerIsWhite ? PAWN_ATTACKS_WHITE : PAWN_ATTACKS_BLACK);
      BitBoard atkByN = ComputeNonSliderAttackUnion(attN, KNIGHT_ATTACKS);
      BitBoard atkByB = ComputeSliderAttackUnion(attB, occupancy, BISHOP_DIRS);
      BitBoard atkByR = ComputeSliderAttackUnion(attR, occupancy, ROOK_DIRS);
      BitBoard atkByQ = ComputeSliderAttackUnion(attQ, occupancy, BISHOP_DIRS) | ComputeSliderAttackUnion(attQ, occupancy, ROOK_DIRS);

      // King: threatened iff attacked at all (king has no lower-value attacker — but check is most-critical)
      BitBoard anyAttack = atkByP | atkByN | atkByB | atkByR | atkByQ;
      ApplyThreatMask(threatened, defK, anyAttack);
      // Queen: threatened by P/N/B/R (anything except Q)
      ApplyThreatMask(threatened, defQ, atkByP | atkByN | atkByB | atkByR);
      // Rook: threatened by P/N/B (cheaper pieces)
      ApplyThreatMask(threatened, defR, atkByP | atkByN | atkByB);
      // Knight/Bishop (~equal value 3): threatened by pawn only
      ApplyThreatMask(threatened, defN, atkByP);
      ApplyThreatMask(threatened, defB, atkByP);
      // Pawn: only pawn attacks pawn (equal trade) — not strictly lower-value attacker. Skip.
    }

    private static BitBoard ComputeNonSliderAttackUnion(BitBoard pieces, BitBoard[] attackTable)
    {
      BitBoard u = 0;
      while (pieces != 0)
      {
        int sq = BitOperations.TrailingZeroCount(pieces);
        pieces &= pieces - 1;
        u |= attackTable[sq];
      }
      return u;
    }

    private static BitBoard ComputeSliderAttackUnion(BitBoard pieces, BitBoard occupancy, (int df, int dr)[] dirs)
    {
      BitBoard u = 0;
      while (pieces != 0)
      {
        int sq = BitOperations.TrailingZeroCount(pieces);
        pieces &= pieces - 1;
        u |= SlidingAttacks(sq, occupancy, dirs);
      }
      return u;
    }

    private static void ApplyThreatMask(Span<byte> threatened, BitBoard defenderPieces, BitBoard threatMask)
    {
      BitBoard hit = defenderPieces & threatMask;
      while (hit != 0)
      {
        int sq = BitOperations.TrailingZeroCount(hit);
        hit &= hit - 1;
        threatened[sq] = 100;
      }
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

    // ============================================================================
    // SEE (Static Exchange Evaluation) helpers were DROPPED 2026-06-01 after ablation
    // showed SEE was redundant with is_threatened + the model's internal reasoning.
    // Removed code: ComputeSEEPerSquare, SEEAt, PieceValueAt, AttackersOfSquare, XRayThrough
    // (preserved in git history, commit prior to V3 cleanup).
    // ============================================================================
  }
}
