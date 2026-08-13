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
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

using BitBoard = System.UInt64;

#endregion

namespace Ceres.Chess.MoveGen
{
  /// <summary>
  /// Constant-time sliding piece attack generation (plus knight/king attack sets
  /// and between/line masks) for the square indexing used throughout MGPosition,
  /// where bit 0 is h1, bit 7 is a1, and bit 63 is a8.
  ///
  /// Whole attack sets are produced with a single table lookup.
  ///
  /// Two index functions are supported over an identically shaped table:
  ///   - PEXT (BMI2), used when the hardware implements it in a few cycles, and
  ///   - classical magic multiply, otherwise. Magics are searched for at startup
  ///     with a fixed-seed PRNG (the published magic constants assume a1 == bit 0,
  ///     so they cannot be reused under this mirrored indexing).
  /// </summary>
  internal static class MGSliderAttacks
  {
    #region Direction tables

    // Directions expressed as (rank delta, index-within-rank delta).
    // Note the index within a rank runs h..a, so "+1" moves toward the a file.
    static readonly (int dRank, int dFile)[] ROOK_DIRS = { (1, 0), (-1, 0), (0, 1), (0, -1) };
    static readonly (int dRank, int dFile)[] BISHOP_DIRS = { (1, 1), (1, -1), (-1, 1), (-1, -1) };

    #endregion

    #region Public precomputed tables

    /// <summary>
    /// Knight attack set from each square.
    /// </summary>
    internal static readonly BitBoard[] KnightAttacks = new BitBoard[64];

    /// <summary>
    /// King attack set from each square.
    /// </summary>
    internal static readonly BitBoard[] KingAttacks = new BitBoard[64];

    /// <summary>
    /// Squares strictly between two squares if they share a rank, file or diagonal (otherwise 0).
    /// </summary>
    internal static readonly BitBoard[] Between = new BitBoard[64 * 64];

    /// <summary>
    /// All squares of the rank, file or diagonal shared by two squares
    /// (including both endpoints), or 0 if they are not aligned.
    /// </summary>
    internal static readonly BitBoard[] Line = new BitBoard[64 * 64];

    #endregion

    #region Magic table

    struct MagicEntry
    {
      public BitBoard Mask;
      public BitBoard Magic;
      public int Offset;
      public int Shift;
    }

    static readonly MagicEntry[] rookMagics = new MagicEntry[64];
    static readonly MagicEntry[] bishopMagics = new MagicEntry[64];
    static readonly BitBoard[] attackTable;

    /// <summary>
    /// True if hardware PEXT is used to index the attack table (rather than a magic multiply).
    /// </summary>
    internal static readonly bool UsesPext;

    const int ROOK_TABLE_SIZE = 102400;
    const int BISHOP_TABLE_SIZE = 5248;

    #endregion

    #region Initialization

    static MGSliderAttacks()
    {
      UsesPext = PextIsFast();

      attackTable = new BitBoard[ROOK_TABLE_SIZE + BISHOP_TABLE_SIZE];

      InitLeaperAttacks();
      InitMagics(rookMagics, ROOK_DIRS, ROOK_MAGICS, 0);
      InitMagics(bishopMagics, BISHOP_DIRS, BISHOP_MAGICS, ROOK_TABLE_SIZE);
      InitBetweenAndLine();
    }


    /// <summary>
    /// Returns whether BMI2 PEXT is available and known to execute quickly.
    ///
    /// AMD implemented PEXT in microcode through Zen 2 (family 0x17), where it costs
    /// roughly 18 cycles and is far slower than a magic multiply; it became a 3 cycle
    /// instruction with Zen 3 (family 0x19).
    /// </summary>
    static bool PextIsFast()
    {
      if (!Bmi2.X64.IsSupported)
      {
        return false;
      }

      if (!X86Base.IsSupported)
      {
        return false;
      }

      (int Eax, int Ebx, int Ecx, int Edx) vendor = X86Base.CpuId(0, 0);
      bool isAMD = vendor.Ebx == 0x6874_7541 && vendor.Edx == 0x6974_6E65 && vendor.Ecx == 0x444D_4163; // "AuthenticAMD"
      if (!isAMD)
      {
        return true;
      }

      int eax = X86Base.CpuId(1, 0).Eax;
      int baseFamily = (eax >> 8) & 0xF;
      int family = baseFamily == 0xF ? baseFamily + ((eax >> 20) & 0xFF) : baseFamily;

      return family >= 0x19; // Zen 3 and later
    }


    static void InitLeaperAttacks()
    {
      ReadOnlySpan<int> knightRank = [2, 2, 1, 1, -1, -1, -2, -2];
      ReadOnlySpan<int> knightFile = [1, -1, 2, -2, 2, -2, 1, -1];

      for (int sq = 0; sq < 64; sq++)
      {
        int r0 = sq / 8;
        int f0 = sq % 8;

        BitBoard n = 0;
        for (int i = 0; i < 8; i++)
        {
          int r = r0 + knightRank[i];
          int f = f0 + knightFile[i];
          if ((uint)r < 8 && (uint)f < 8)
          {
            n |= 1UL << (r * 8 + f);
          }
        }
        KnightAttacks[sq] = n;

        BitBoard k = 0;
        for (int dr = -1; dr <= 1; dr++)
        {
          for (int df = -1; df <= 1; df++)
          {
            if (dr == 0 && df == 0)
            {
              continue;
            }
            int r = r0 + dr;
            int f = f0 + df;
            if ((uint)r < 8 && (uint)f < 8)
            {
              k |= 1UL << (r * 8 + f);
            }
          }
        }
        KingAttacks[sq] = k;
      }
    }


    /// <summary>
    /// Slow reference generation of a slider attack set (used only to build the tables).
    /// </summary>
    static BitBoard SlowAttacks(int sq, BitBoard occupied, (int dRank, int dFile)[] dirs)
    {
      BitBoard result = 0;
      int r0 = sq / 8;
      int f0 = sq % 8;

      foreach ((int dRank, int dFile) in dirs)
      {
        int r = r0 + dRank;
        int f = f0 + dFile;
        while ((uint)r < 8 && (uint)f < 8)
        {
          int s = r * 8 + f;
          result |= 1UL << s;
          if ((occupied & (1UL << s)) != 0)
          {
            break;
          }
          r += dRank;
          f += dFile;
        }
      }

      return result;
    }


    /// <summary>
    /// Relevant occupancy mask: the attack set on an empty board minus the final
    /// square in each direction (whose occupancy cannot affect the result).
    /// </summary>
    static BitBoard RelevantOccupancy(int sq, (int dRank, int dFile)[] dirs)
    {
      BitBoard result = 0;
      int r0 = sq / 8;
      int f0 = sq % 8;

      foreach ((int dRank, int dFile) in dirs)
      {
        int r = r0 + dRank;
        int f = f0 + dFile;
        while ((uint)(r + dRank) < 8 && (uint)(f + dFile) < 8)
        {
          result |= 1UL << (r * 8 + f);
          r += dRank;
          f += dFile;
        }
      }

      return result;
    }


    /// <summary>
    /// Per-rank PRNG seeds, used only if a precomputed magic below ever fails to verify.
    /// </summary>
    static readonly ulong[] MAGIC_SEEDS = { 728, 10316, 55013, 32803, 12281, 15100, 16645, 255 };


    // Magic constants for this board's square indexing (bit 0 == h1). The published
    // constants in the literature assume a1 == bit 0 and do not carry over, so these
    // were found by the search below and then frozen to avoid paying for it at startup.
    // They are verified while the table is built; if verification ever fails the
    // search runs for that square, so a bad constant cannot produce wrong attacks.

    static readonly BitBoard[] ROOK_MAGICS =
    {
      0x0A80004000801220UL, 0x0040001000200040UL, 0x0A00108008402202UL, 0x9580100008028004UL,
      0x0100040801001002UL, 0x1300124400110008UL, 0x4400410082280410UL, 0x010000E208448B00UL,
      0x1085800240088020UL, 0x40814000A0005001UL, 0x0342004200802010UL, 0x8401000A21021002UL,
      0x2423000800051100UL, 0x4200808002000400UL, 0x4203000100020004UL, 0x6002001060820504UL,
      0x028004400220024AUL, 0x5040002010002802UL, 0x9008410010200100UL, 0x8101010020100008UL,
      0x5508008080080400UL, 0x0004008080040200UL, 0x0008010100020004UL, 0x0202020024008041UL,
      0x0084209980024000UL, 0x2400200240100040UL, 0x7128401100200102UL, 0x0140082200401200UL,
      0x0408040080800800UL, 0x0910040801102040UL, 0x0060020400100801UL, 0x024004820004410CUL,
      0x0800804004800420UL, 0x0000400080802008UL, 0x4010002000808010UL, 0x0888801000800800UL,
      0x0D18040080800802UL, 0x0006000401010008UL, 0x0032000102000408UL, 0x00C0800042800700UL,
      0x0080004020004001UL, 0x0000400089090020UL, 0x0000108042020024UL, 0x0000210010030008UL,
      0x800D040008008080UL, 0xA000040002008080UL, 0x4021000200010104UL, 0x4108040040820001UL,
      0xC90C801240002080UL, 0x0052002041008200UL, 0x0001021248200100UL, 0x0000100180180180UL,
      0x00000A0012A01A00UL, 0x2C52020004008080UL, 0x0008D82110021400UL, 0x0024800100004080UL,
      0x0010104100208001UL, 0x0001102082410A02UL, 0x000F000842200011UL, 0x5352041001000921UL,
      0x2003000800020411UL, 0x0601000804000201UL, 0x2800284081021004UL, 0x0800040080410022UL,
    };

    static readonly BitBoard[] BISHOP_MAGICS =
    {
      0x004002020A003102UL, 0x3058114410820108UL, 0x08A2020042000000UL, 0x0008049508000401UL,
      0x1001104181080082UL, 0x1102021004000820UL, 0x80AC249808280003UL, 0x0000404050101080UL,
      0x1000200202480120UL, 0x2040080108620143UL, 0x0004080809282202UL, 0x40005444018808A1UL,
      0x2060640504000004UL, 0x0108021804040000UL, 0x40000104107C0405UL, 0x0008020D00884400UL,
      0x4011041856300C00UL, 0x008404D084208400UL, 0x600810100044400AUL, 0x0004000802400C00UL,
      0x0000804408A00141UL, 0x5102000022100200UL, 0x2340822202012100UL, 0x262A800300415010UL,
      0x0430300118021080UL, 0x0042700020010A00UL, 0x0808500008002040UL, 0x0004004034010002UL,
      0x0050040004802100UL, 0x000109004A00412BUL, 0x0002238004041100UL, 0x0002002000412800UL,
      0x2010042000060800UL, 0xC044012C00185010UL, 0x2042032400420808UL, 0x2480208020080200UL,
      0x0040008208210100UL, 0x0020808205090100UL, 0x8801020200809800UL, 0x04008A2040020108UL,
      0x1284026084011020UL, 0x1101010120009040UL, 0x000C08C248011000UL, 0x0100002204220800UL,
      0x000204090A040400UL, 0x0020121042000041UL, 0x008421CA02000408UL, 0x10240400A2040020UL,
      0x806480907010C00AUL, 0x0101044910080198UL, 0x0060010080900024UL, 0x0002020442022201UL,
      0x0011E02082440010UL, 0x0002485090108000UL, 0x0120048102140100UL, 0x0421820400498040UL,
      0x0021088209014000UL, 0x4220344628040201UL, 0x4400020024020822UL, 0x24C0000100840420UL,
      0x1211080005050401UL, 0x00800820121A2200UL, 0x000041104408A080UL, 0x0002301006118824UL,
    };


    static void InitMagics(MagicEntry[] magics, (int dRank, int dFile)[] dirs, BitBoard[] precomputed, int baseOffset)
    {
      BitBoard[] occupancies = new BitBoard[4096];
      BitBoard[] reference = new BitBoard[4096];
      int[] epoch = new int[4096];
      int currentEpoch = 0;
      int offset = baseOffset;

      ulong rngState = 0;

      for (int sq = 0; sq < 64; sq++)
      {
        rngState = 0;

        BitBoard mask = RelevantOccupancy(sq, dirs);
        int bits = BitOperations.PopCount(mask);
        int size = 1 << bits;

        magics[sq].Mask = mask;
        magics[sq].Offset = offset;
        magics[sq].Shift = 64 - bits;

        // Enumerate every subset of the mask (Carry-Rippler).
        BitBoard occ = 0;
        int n = 0;
        do
        {
          occupancies[n] = occ;
          reference[n] = SlowAttacks(sq, occ, dirs);
          n++;
          occ = (occ - mask) & mask;
        } while (occ != 0);

        if (UsesPext)
        {
          for (int i = 0; i < n; i++)
          {
            attackTable[offset + (int)Bmi2.X64.ParallelBitExtract(occupancies[i], mask)] = reference[i];
          }
        }
        else
        {
          // Try the frozen constant first; fall back to searching if it does not verify.
          // A magic is acceptable when every occupancy subset maps to a slot holding the
          // correct attack set (constructive collisions are allowed).
          BitBoard candidate = precomputed[sq];

          while (true)
          {
            currentEpoch++;
            bool ok = true;
            for (int i = 0; i < n; i++)
            {
              int idx = (int)((occupancies[i] * candidate) >> (64 - bits));
              if (epoch[idx] != currentEpoch)
              {
                epoch[idx] = currentEpoch;
                attackTable[offset + idx] = reference[i];
              }
              else if (attackTable[offset + idx] != reference[i])
              {
                ok = false;
                break;
              }
            }

            if (ok)
            {
              magics[sq].Magic = candidate;
              break;
            }

            if (rngState == 0)
            {
              rngState = MAGIC_SEEDS[sq / 8] + (ulong)(baseOffset + sq) * 0x9E3779B97F4A7C15UL;
            }

            do
            {
              candidate = SparseRandom(ref rngState);
            } while (BitOperations.PopCount((mask * candidate) & 0xFF00000000000000UL) < 6);
          }
        }

        offset += size;
      }
    }


    static ulong NextRandom(ref ulong s)
    {
      s ^= s >> 12;
      s ^= s << 25;
      s ^= s >> 27;
      return s * 2685821657736338717UL;
    }


    static ulong SparseRandom(ref ulong s) => NextRandom(ref s) & NextRandom(ref s) & NextRandom(ref s);


    static void InitBetweenAndLine()
    {
      for (int a = 0; a < 64; a++)
      {
        int ra = a / 8;
        int fa = a % 8;

        for (int b = 0; b < 64; b++)
        {
          if (a == b)
          {
            continue;
          }

          int rb = b / 8;
          int fb = b % 8;

          int dr = Math.Sign(rb - ra);
          int df = Math.Sign(fb - fa);

          bool aligned = (ra == rb) || (fa == fb) || (Math.Abs(rb - ra) == Math.Abs(fb - fa));
          if (!aligned)
          {
            continue;
          }

          // Squares strictly between.
          BitBoard between = 0;
          int r = ra + dr;
          int f = fa + df;
          while (r != rb || f != fb)
          {
            between |= 1UL << (r * 8 + f);
            r += dr;
            f += df;
          }
          Between[a * 64 + b] = between;

          // Full line through both squares, extended to the board edges.
          BitBoard line = (1UL << a) | (1UL << b) | between;
          r = ra - dr;
          f = fa - df;
          while ((uint)r < 8 && (uint)f < 8)
          {
            line |= 1UL << (r * 8 + f);
            r -= dr;
            f -= df;
          }
          r = rb + dr;
          f = fb + df;
          while ((uint)r < 8 && (uint)f < 8)
          {
            line |= 1UL << (r * 8 + f);
            r += dr;
            f += df;
          }
          Line[a * 64 + b] = line;
        }
      }
    }

    #endregion

    #region Attack lookups

    static BitBoard Probe(MagicEntry[] magics, int sq, BitBoard occupied)
    {
      ref MagicEntry m = ref magics[sq];
      int index;
      if (UsesPext)
      {
        index = m.Offset + (int)Bmi2.X64.ParallelBitExtract(occupied, m.Mask);
      }
      else
      {
        index = m.Offset + (int)(((occupied & m.Mask) * m.Magic) >> m.Shift);
      }

      return attackTable[index];
    }


    /// <summary>
    /// Returns all squares attacked by a rook on the given square for the given occupancy.
    /// Note that the occupancy passed must not include en passant marker squares,
    /// which are physically empty and therefore do not block sliders.
    /// </summary>
    internal static BitBoard Rook(int sq, BitBoard occupied) => Probe(rookMagics, sq, occupied);


    /// <summary>
    /// Returns all squares attacked by a bishop on the given square for the given occupancy.
    /// </summary>
    internal static BitBoard Bishop(int sq, BitBoard occupied) => Probe(bishopMagics, sq, occupied);


    /// <summary>
    /// Returns all squares attacked by a queen on the given square for the given occupancy.
    /// </summary>
    internal static BitBoard Queen(int sq, BitBoard occupied) => Rook(sq, occupied) | Bishop(sq, occupied);


    /// <summary>
    /// Squares strictly between two squares (0 if they are not aligned).
    /// </summary>
    internal static BitBoard BetweenBB(int a, int b) => Between[(a << 6) + b];


    /// <summary>
    /// The full rank/file/diagonal shared by two squares (0 if they are not aligned).
    /// </summary>
    internal static BitBoard LineBB(int a, int b) => Line[(a << 6) + b];


    /// <summary>
    /// Emits the magic constants currently in use as C# source, so that a newly
    /// searched set can be pasted back into ROOK_MAGICS / BISHOP_MAGICS above.
    /// Only meaningful when running with the magic index path selected.
    /// </summary>
    internal static string DumpMagics()
    {
      System.Text.StringBuilder sb = new System.Text.StringBuilder();
      void Emit(string name, MagicEntry[] magics)
      {
        sb.AppendLine($"    static readonly BitBoard[] {name} =");
        sb.AppendLine("    {");
        for (int sq = 0; sq < 64; sq += 4)
        {
          sb.Append("     ");
          for (int i = 0; i < 4; i++)
          {
            sb.Append($" 0x{magics[sq + i].Magic:X16}UL,");
          }
          sb.AppendLine();
        }
        sb.AppendLine("    };");
      }

      Emit("ROOK_MAGICS", rookMagics);
      Emit("BISHOP_MAGICS", bishopMagics);
      return sb.ToString();
    }

    #endregion
  }
}
