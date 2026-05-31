// Phase 1 sanity + Phase 2 equality tests for PerSquareAttacks.
// Phase 1: starting position totals 38/38, symmetry, e4 spot check.
// Phase 2: bit-exact equality with Python (aug_features.py / aug_features_dump_for_fen.py)
//          on 25 hand-picked FENs (start, mid-game, EP, castling, endgame, mate).

using System;
using System.IO;
using System.Runtime.InteropServices;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using Ceres.Chess.PositionDataInfo;

namespace AugFeatSanity;

class Program
{
  static int Main(string[] args)
  {
    // Phase 0: confirm v3 layout constants are in effect.
    int sqRecSize    = Marshal.SizeOf<TPGSquareRecord>();
    int sq64Size     = Marshal.SizeOf<TPGSquareRecord64>();
    int tpgRecSize   = Marshal.SizeOf<TPGRecord>();
    Console.WriteLine($"Phase 0: layout sanity");
    Console.WriteLine($"  sizeof(TPGSquareRecord)            = {sqRecSize}    (expected 140 with V3)");
    Console.WriteLine($"  sizeof(TPGSquareRecord64)          = {sq64Size}   (expected 64 * 140 = 8960)");
    Console.WriteLine($"  sizeof(TPGRecord)                  = {tpgRecSize}");
    Console.WriteLine($"  TPGRecord.TOTAL_BYTES (const)      = {TPGRecord.TOTAL_BYTES}");
    Console.WriteLine($"  TPGRecord.BYTES_PER_SQUARE_RECORD  = {TPGRecord.BYTES_PER_SQUARE_RECORD}");
    Console.WriteLine($"  USE_V2={TPGRecord.USE_V2_TPG_RECORD}, USE_V3={TPGRecord.USE_V3_TPG_RECORD}");
    int expectedSq = TPGRecord.USE_V3_TPG_RECORD ? 140 : 137;
    if (sqRecSize != expectedSq || TPGRecord.BYTES_PER_SQUARE_RECORD != expectedSq)
    {
      Console.WriteLine($"  FAIL: expected TPGSquareRecord size {expectedSq}");
      return 99;
    }
    if (tpgRecSize != TPGRecord.TOTAL_BYTES)
    {
      Console.WriteLine($"  FAIL: sizeof(TPGRecord)={tpgRecSize} != TOTAL_BYTES={TPGRecord.TOTAL_BYTES} (off by {tpgRecSize - TPGRecord.TOTAL_BYTES})");
      Console.WriteLine($"  This means the TPG file format const is out of sync with the actual struct — writer will corrupt files.");
      return 98;
    }
    Console.WriteLine($"  PASS (v3 layout: 140 bytes/square, TPGRecord {tpgRecSize} bytes matches TOTAL_BYTES)");
    Console.WriteLine();

    // Test 1: starting position.
    var startPos = MGChessPositionConverter.MGChessPositionFromFEN(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    Span<byte> w = stackalloc byte[64];
    Span<byte> b = stackalloc byte[64];
    PerSquareAttacks.Compute(in startPos, w, b);

    int wTotal = 0, bTotal = 0;
    for (int i = 0; i < 64; i++) { wTotal += w[i]; bTotal += b[i]; }

    Console.WriteLine($"Test 1: starting position");
    Console.WriteLine($"  white total attackers = {wTotal}  (python-chess truth: 38)");
    Console.WriteLine($"  black total attackers = {bTotal}  (python-chess truth: 38)");
    Console.WriteLine($"  symmetric (W==B): {wTotal == bTotal}");

    if (wTotal != 38 || bTotal != 38)
    {
      Console.WriteLine("FAIL: starting-position totals don't match python-chess");
      PrintBoard("white attackers", w);
      PrintBoard("black attackers", b);
      return 1;
    }

    // Test 2: rank-mirror symmetry of starting position.
    // For each rank r (0..7), white_attackers on rank r must equal black_attackers on rank (7 - r).
    int mismatches = 0;
    for (int r = 0; r < 8; r++)
      for (int f = 0; f < 8; f++)
      {
        int sqW = r * 8 + f;
        int sqB = (7 - r) * 8 + f;
        if (w[sqW] != b[sqB])
        {
          Console.WriteLine($"  asymmetry: white[{Sq(sqW)}]={w[sqW]} != black[{Sq(sqB)}]={b[sqB]}");
          mismatches++;
        }
      }
    if (mismatches > 0)
    {
      Console.WriteLine($"FAIL: {mismatches} symmetry violations");
      PrintBoard("white attackers", w);
      PrintBoard("black attackers", b);
      return 2;
    }
    Console.WriteLine($"  passes rank-mirror symmetry test");

    // Test 3: spot-check a known square (e4 in starting pos = 28).
    // In starting position, e4 (sq 28) is attacked by:
    //   white: Ng1 doesn't reach e4; Nb1 doesn't reach e4; pawn d2/f2 attack... wait
    //   actually let me think: white pawn on e2 sees d3, f3. Doesn't attack e4.
    //   white knight Nb1 attacks a3,c3. Ng1 attacks f3,h3. Neither attacks e4.
    //   So no white pieces attack e4. Expected: 0.
    // Black: e7 pawn attacks d6, f6. Nothing reaches e4.
    // Expected white_attackers[e4] = 0, black_attackers[e4] = 0.
    int e4 = 28;  // file 4 (e), rank 3 (rank 4) → 3*8+4 = 28
    Console.WriteLine($"Test 3: e4 (sq 28) attackers in starting position");
    Console.WriteLine($"  white = {w[e4]} (expected 0)");
    Console.WriteLine($"  black = {b[e4]} (expected 0)");
    if (w[e4] != 0 || b[e4] != 0)
    {
      Console.WriteLine("FAIL: e4 attackers should be 0/0 in starting position");
      return 3;
    }

    // Test 4: spot-check a4 (sq 24) in starting pos.
    // White Nb1 attacks a3 not a4. White b2 pawn doesn't reach a4. So 0 white attackers.
    // Black has nothing reaching a4 either. Expected 0/0.
    // Better spot-check: e1 (white king) — attacked by white d1 queen? Let me think:
    //   - White king on e1: attacked by D1 (Q), e2 (P), f1 (B), no, attacked counts pieces of color X
    //     attacking sq. White king on e1: who from white attacks e1? d1 queen reaches e1 (diagonal),
    //     f1 bishop reaches e1 (diagonal), d2 pawn does NOT attack e1 (pawns attack forward not back).
    //     Wait pawn on d2 attacks c3 and e3. So no.
    //     d1 queen does attack e1 (same rank). f1 bishop attacks e1? f1 bishop on diagonal e2-d3...
    //     no, f1's diagonals go up-left (e2, d3...) and up-right (g2, h3). Doesn't attack e1.
    //     Actually a king-adjacent square is attacked by the adjacent pieces.
    //     e1 (own square) attackers from white:
    //       - d1 Q attacks e1? Q on d1 has rays in all 8 directions. East reaches e1 immediately. Yes.
    //       - f1 B? f1 bishop has diagonals. e2 (up-left), g2 (up-right), no e1.
    //       - d2 P? attacks c3, e3, not e1.
    //     So white_attackers[e1] = 1 (just d1 queen).
    // Per python-chess truth check this should be confirmable in the selftest.
    // For now, trust the symmetric + total = 38 + e4 = 0/0 trio.

    Console.WriteLine();
    Console.WriteLine("Phase 1 tests PASSED");
    Console.WriteLine();

    // ---------- Phase 2: Python ↔ C# bit-exact equality test ---------------
    string fensPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "equality_fens.txt");
    string bytesPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "aug_bytes_python.bin");
    if (!File.Exists(fensPath) || !File.Exists(bytesPath))
    {
      Console.WriteLine($"Phase 2 SKIPPED: missing {(File.Exists(fensPath) ? "" : "fens.txt ")} {(File.Exists(bytesPath) ? "" : "aug_bytes_python.bin")}");
      Console.WriteLine($"  Expected paths: {fensPath}  |  {bytesPath}");
      return 0;
    }

    int phase2Result = RunPhase2EqualityTest(fensPath, bytesPath);
    if (phase2Result != 0) return phase2Result;

    int phase3Result = RunPhase3TPGBakeInTest(fensPath);
    if (phase3Result != 0) return phase3Result;

    Console.WriteLine();
    Console.WriteLine("ALL TESTS PASSED (Phase 1 + Phase 2 + Phase 3 v3-bake-in)");
    Console.WriteLine();
    return 0;
  }

  /// <summary>
  /// Phase 3: end-to-end v3-layout test. For each FEN, build TPGSquareRecord[64] via
  /// the production WritePosPieces() path, extract augFeatureBytes from each square,
  /// and compare against the same bytes computed directly via PerSquareAttacks on the
  /// real position + the canonical 180-rotation mapping.
  ///
  /// Per-channel byte derivation uses RAW counts (not the oracle's pre-quantized bytes
  /// — quantization is lossy on count >> byte >> count round-trip). This isolates
  /// the load-bearing assertion: WritePosPieces correctly orients aug bytes into TPG
  /// slots regardless of side-to-move. The Python-equivalence of the byte VALUES
  /// themselves is already established by Phase 2.
  /// </summary>
  static int RunPhase3TPGBakeInTest(string fensPath)
  {
    Console.WriteLine($"Phase 3: v3 TPG bake-in equality test (WritePosPieces vs PerSquareAttacks+orientation)");

    var fens = new System.Collections.Generic.List<string>();
    foreach (var line in File.ReadAllLines(fensPath))
    {
      var l = line.Trim();
      if (l.Length == 0 || l.StartsWith("#")) continue;
      fens.Add(l);
    }

    int totalMismatches = 0;
    for (int fenIdx = 0; fenIdx < fens.Count; fenIdx++)
    {
      string fen = fens[fenIdx];
      Position pos = Position.FromFEN(fen);
      bool weAreWhite = pos.MiscInfo.SideToMove == SideType.White;

      // Compute expected attack counts directly on the real position (raw bytes 0..8).
      MGPosition mgPos = MGPosition.FromPosition(in pos);
      Span<byte> wAtt = stackalloc byte[64];
      Span<byte> bAtt = stackalloc byte[64];
      PerSquareAttacks.Compute(in mgPos, wAtt, bAtt);

      // Run the production v3 generator path.
      Span<TPGSquareRecord> squareRecords = stackalloc TPGSquareRecord[64];
      // history params unused for aug correctness; pass pos placeholder
      TPGSquareRecord.WritePosPieces(in pos, in pos, in pos, in pos, in pos, in pos, in pos, in pos,
                                     squareRecords, Span<byte>.Empty, false, 0f, 0f);

      int mismatchesThisFen = 0;
      for (int tpgIdx = 0; tpgIdx < 64; tpgIdx++)
      {
        // Per WritePosPieces: W-to-move places real-i at TPG-i; B-to-move places real-i at TPG-(63-i).
        // So inverse: TPG idx t → real square (W: t, B: 63-t).
        int realSq = weAreWhite ? tpgIdx : (63 - tpgIdx);
        int ourCount = weAreWhite ? wAtt[realSq] : bAtt[realSq];
        int oppCount = weAreWhite ? bAtt[realSq] : wAtt[realSq];

        byte expectedOur = (byte)(ourCount * 100 / 8);
        byte expectedOpp = (byte)(oppCount * 100 / 8);
        byte expectedNet = (byte)((ourCount - oppCount + 8) * 100 / 16);

        ReadOnlySpan<byte> actualAug = squareRecords[tpgIdx].AuxFeatureBytesReadOnly;
        if (actualAug[0] != expectedOur || actualAug[1] != expectedOpp || actualAug[2] != expectedNet)
        {
          if (mismatchesThisFen < 3)
          {
            Console.WriteLine($"  MISMATCH FEN[{fenIdx}] tpg={tpgIdx} (real={realSq}, weAreWhite={weAreWhite}):");
            Console.WriteLine($"    raw   our={ourCount} opp={oppCount}");
            Console.WriteLine($"    expected our={expectedOur} opp={expectedOpp} net={expectedNet}");
            Console.WriteLine($"    actual   our={actualAug[0]} opp={actualAug[1]} net={actualAug[2]}");
          }
          mismatchesThisFen++;
        }
      }
      if (mismatchesThisFen > 0)
      {
        Console.WriteLine($"  FEN[{fenIdx}] = {fen}: {mismatchesThisFen} byte mismatches");
        totalMismatches += mismatchesThisFen;
      }
    }

    Console.WriteLine($"  evaluated {fens.Count} FENs * 64 tpg-squares * 3 channels = {fens.Count * 192} byte comparisons");
    if (totalMismatches == 0)
    {
      Console.WriteLine($"  PASS: WritePosPieces orients aug bytes correctly for both W-to-move and B-to-move");
      return 0;
    }
    Console.WriteLine($"  FAIL: {totalMismatches} mismatches");
    return 31;
  }

  /// <summary>
  /// Phase 2: load FENs + Python-computed bytes, compute C# bytes, byte-for-byte compare.
  /// Returns 0 on success, nonzero on first mismatch.
  /// </summary>
  static int RunPhase2EqualityTest(string fensPath, string bytesPath)
  {
    Console.WriteLine($"Phase 2: Python ↔ C# equality test");
    Console.WriteLine($"  FENs:  {fensPath}");
    Console.WriteLine($"  bytes: {bytesPath}");

    var fens = new System.Collections.Generic.List<string>();
    foreach (var line in File.ReadAllLines(fensPath))
    {
      var l = line.Trim();
      if (l.Length == 0 || l.StartsWith("#")) continue;
      fens.Add(l);
    }

    byte[] pythonBytes = File.ReadAllBytes(bytesPath);
    int nFensInFile = BitConverter.ToInt32(pythonBytes, 0);
    if (nFensInFile != fens.Count)
    {
      Console.WriteLine($"  FAIL: header says {nFensInFile} FENs but FEN file has {fens.Count}");
      return 10;
    }

    int totalMismatches = 0;
    for (int i = 0; i < fens.Count; i++)
    {
      string fen = fens[i];
      MGPosition pos;
      try { pos = MGChessPositionConverter.MGChessPositionFromFEN(fen); }
      catch (Exception ex)
      {
        Console.WriteLine($"  FAIL FEN[{i}]: parse error: {ex.Message}");
        return 11;
      }

      Span<byte> wCount = stackalloc byte[64];
      Span<byte> bCount = stackalloc byte[64];
      PerSquareAttacks.Compute(in pos, wCount, bCount);

      // Build the 192-byte C# representation matching aug_features.py encoding:
      //   byte[sq*3 + 0] = our (WHITE) count * 100 / 8
      //   byte[sq*3 + 1] = opp (BLACK) count * 100 / 8
      //   byte[sq*3 + 2] = (our - opp + 8) * 100 / 16
      // NOTE: dump script writes REAL-board colors (WHITE in oracle = WHITE in C#).
      // No us-to-move flip here — that's a TPG-layout concern, not a feature-value concern.
      Span<byte> csharpBytes = stackalloc byte[64 * 3];
      for (int sq = 0; sq < 64; sq++)
      {
        int w = wCount[sq];
        int b = bCount[sq];
        csharpBytes[sq * 3 + 0] = (byte)((w * 100) / 8);
        csharpBytes[sq * 3 + 1] = (byte)((b * 100) / 8);
        csharpBytes[sq * 3 + 2] = (byte)(((w - b + 8) * 100) / 16);
      }

      // Compare byte-for-byte against Python reference (offset 4 + i*192).
      int offset = 4 + i * 192;
      int mismatchesThisFen = 0;
      for (int k = 0; k < 192; k++)
      {
        if (pythonBytes[offset + k] != csharpBytes[k])
        {
          if (mismatchesThisFen < 3) // print first 3 mismatches per FEN to aid debug
          {
            int sq = k / 3, channel = k % 3;
            string chName = channel == 0 ? "our" : channel == 1 ? "opp" : "net";
            Console.WriteLine($"  MISMATCH FEN[{i}] sq={Sq(sq)} ch={chName}: python={pythonBytes[offset + k]} csharp={csharpBytes[k]}");
          }
          mismatchesThisFen++;
        }
      }
      if (mismatchesThisFen > 0)
      {
        Console.WriteLine($"  FEN[{i}] = {fen}");
        Console.WriteLine($"    {mismatchesThisFen} byte mismatches");
        totalMismatches += mismatchesThisFen;
      }
    }

    Console.WriteLine($"  evaluated {fens.Count} FENs × 192 bytes = {fens.Count * 192} byte comparisons");
    if (totalMismatches == 0)
    {
      Console.WriteLine($"  PASS: zero byte mismatches");
      return 0;
    }
    else
    {
      Console.WriteLine($"  FAIL: {totalMismatches} total byte mismatches across {fens.Count} FENs");
      return 20;
    }
  }

  static string Sq(int sqIdx)
  {
    char f = (char)('a' + (sqIdx & 7));
    char r = (char)('1' + (sqIdx >> 3));
    return $"{f}{r}";
  }

  static void PrintBoard(string label, Span<byte> b)
  {
    Console.WriteLine($"  {label}:");
    for (int r = 7; r >= 0; r--)
    {
      Console.Write($"  {r + 1}: ");
      for (int f = 0; f < 8; f++) Console.Write($"{b[r * 8 + f]} ");
      Console.WriteLine();
    }
    Console.WriteLine("     a b c d e f g h");
  }
}
