#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region License

/*
License Note

This code originated from Github repository from Judd Niemann
and is licensed with the MIT License.

This version is modified by David Elliott, including a translation to C# and
some moderate modifications to improve performance and modularity.

Subsequent enhancements were made by Claude Opus 5 to integrate slider attacks.
*/

/*

MIT License

Copyright(c) 2016-2017 Judd Niemann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#endregion

#region Using directives

using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;

#endregion

using BitBoard = System.UInt64;

namespace Ceres.Chess.MoveGen
{
  /// <summary>
  /// Move generator.
  ///
  /// Legality is decided from a small amount of state computed once per position
  /// (see <see cref="MGGenContext"/>): the enemy attack map, the checkers bitboard
  /// and the set of pinned pieces. Given those, an ordinary move is legal exactly
  /// when its destination lies in the check-evasion mask and, if the moving piece
  /// is pinned, on the pin ray. Only king moves, castling and en passant captures
  /// need anything more.
  ///
  /// Note that we don't set the check flag (_FLAG_CHECKS_IN_MOVE_GENERATION_),
  /// for two reasons:
  ///   - it's expensive (slows move generation by 30%), and
  ///   - the code does not yet properly handle computing this for promotions
  /// </summary>
  public static partial class MGMoveGen
  {
    #region Per-position legality context

    /// <summary>
    /// State computed once per position which makes the legality of almost every
    /// candidate move decidable with a pair of mask tests, rather than by
    /// materializing a trial board and running full check detection per candidate.
    /// </summary>
    internal struct MGGenContext
    {
      /// <summary>
      /// Occupied squares excluding en passant markers, which sit on physically
      /// empty squares and therefore do not block sliding pieces.
      /// </summary>
      internal BitBoard OccupiedNoEP;

      /// <summary>
      /// Squares attacked by the side not to move, computed with the moving side's
      /// king removed from the occupancy (so a king cannot retreat along a checking ray).
      /// </summary>
      internal BitBoard DangerMap;

      /// <summary>
      /// Non-sliding (pawn, knight, king) portion of DangerMap, retained so that the
      /// map can be cheaply rebuilt for a different occupancy during castling tests.
      /// </summary>
      internal BitBoard DangerNonSliders;

      /// <summary>
      /// Enemy bishops and queens.
      /// </summary>
      internal BitBoard EnemyDiagonal;

      /// <summary>
      /// Enemy rooks and queens.
      /// </summary>
      internal BitBoard EnemyStraight;

      /// <summary>
      /// Squares on which a non-king move must land: all ones when not in check,
      /// the checker plus any squares between it and the king when in single check,
      /// and zero when in double check (only the king may move).
      /// </summary>
      internal BitBoard CheckMask;

      /// <summary>
      /// Own pieces pinned against own king. A pinned piece may only move along
      /// the line joining the king and the pinning slider.
      /// </summary>
      internal BitBoard Pinned;

      /// <summary>
      /// Bitboard of the moving side's king (zero if the position has no such king).
      /// </summary>
      internal BitBoard KingBB;

      /// <summary>
      /// Square index of the moving side's king (meaningless if KingBB is zero).
      /// </summary>
      internal int KingSq;


      /// <summary>
      /// Returns whether an ordinary (non-king, non-castling, non-en-passant) move
      /// from one square to another leaves the moving side's king safe.
      /// </summary>
      internal readonly bool OrdinaryMoveIsLegal(int fromSquare, BitBoard to)
      {
        if ((CheckMask & to) == 0)
        {
          return false;
        }

        // A pinned piece stays legal only while it remains on the king/pinner line.
        // (The line through the king and any pinned piece is the pin line itself.)
        return (Pinned & (1UL << fromSquare)) == 0
            || (MGSliderAttacks.LineBB(KingSq, fromSquare) & to) != 0;
      }
    }


    /// <summary>
    /// Computes the per-position legality context for the side to move.
    /// </summary>
    static void BuildContext(in MGPosition P, bool blackToMove, out MGGenContext ctx)
    {
      BitBoard occupied = P.A | P.B | P.C;
      BitBoard epSquares = P.A & P.B & ~P.C;
      BitBoard occNoEP = occupied & ~epSquares;

      // Split out the enemy planes. En passant markers and the enemy king fall out
      // of every piece category below (markers have A and B set but not C; the king
      // has A, B and C set), so no explicit masking of them is required.
      BitBoard eA, eB, eC, ourKing, ourOccupied;
      if (blackToMove)
      {
        ourKing = P.A & P.B & P.C & P.D;
        ourOccupied = occNoEP & P.D;
        eA = P.A & ~P.D;
        eB = P.B & ~P.D;
        eC = P.C & ~P.D;
      }
      else
      {
        ourKing = P.A & P.B & P.C & ~P.D;
        ourOccupied = occNoEP & ~P.D;
        eA = P.A & P.D;
        eB = P.B & P.D;
        eC = P.C & P.D;
      }

      BitBoard eAnotB = eA & ~eB;
      BitBoard ePawns = eAnotB & ~eC;
      BitBoard eKnights = eAnotB & eC;
      BitBoard eKing = eA & eB & eC;
      BitBoard eDiagonal = eB & ~eA;  // bishops and queens
      BitBoard eStraight = eC & ~eA;  // rooks and queens

      ctx.OccupiedNoEP = occNoEP;
      ctx.EnemyDiagonal = eDiagonal;
      ctx.EnemyStraight = eStraight;
      ctx.KingBB = ourKing;

      // Attacks which do not depend on occupancy.
      BitBoard dangerNonSliders = (blackToMove ? MGMoveGenFillFunctions.MoveUpLeftRightSingle(ePawns)
                                               : MGMoveGenFillFunctions.MoveDownLeftRightSingle(ePawns))
                                | MGMoveGenFillFunctions.FillKingAttacks(eKing)
                                | MGMoveGenFillFunctions.FillKnightAttacks(eKnights);
      ctx.DangerNonSliders = dangerNonSliders;

      // The king is excluded from the occupancy used for enemy sliders so that squares
      // "behind" the king along a checking ray are correctly marked as attacked.
      ctx.DangerMap = dangerNonSliders | SliderAttackSet(eDiagonal, eStraight, occNoEP & ~ourKing);

      if (ourKing == 0)
      {
        // Degenerate position with no king for the side to move. Preserve the previous
        // behavior, under which check detection always reported "not in check".
        ctx.KingSq = 0;
        ctx.CheckMask = ~0UL;
        ctx.Pinned = 0;
        return;
      }

      int kingSq = BitOperations.TrailingZeroCount(ourKing);
      ctx.KingSq = kingSq;

      // Pieces currently attacking the king.
      BitBoard checkers = (MGSliderAttacks.KnightAttacks[kingSq] & eKnights)
                        | (MGSliderAttacks.Bishop(kingSq, occNoEP) & eDiagonal)
                        | (MGSliderAttacks.Rook(kingSq, occNoEP) & eStraight)
                        | ((blackToMove ? MGMoveGenFillFunctions.MoveDownLeftRightSingle(ourKing)
                                        : MGMoveGenFillFunctions.MoveUpLeftRightSingle(ourKing)) & ePawns);

      if (checkers == 0)
      {
        ctx.CheckMask = ~0UL;
      }
      else if ((checkers & (checkers - 1)) == 0)
      {
        int checkerSq = BitOperations.TrailingZeroCount(checkers);
        ctx.CheckMask = MGSliderAttacks.BetweenBB(kingSq, checkerSq) | checkers;
      }
      else
      {
        ctx.CheckMask = 0; // double check: only the king may move
      }

      // Enemy sliders which would attack the king on an otherwise empty board, and
      // which have exactly one piece in the way, pin that piece.
      BitBoard snipers = (MGSliderAttacks.Rook(kingSq, 0) & eStraight)
                       | (MGSliderAttacks.Bishop(kingSq, 0) & eDiagonal);
      BitBoard pinned = 0;
      while (snipers != 0)
      {
        int sniperSq = BitOperations.TrailingZeroCount(snipers);
        snipers &= snipers - 1;

        BitBoard blockers = MGSliderAttacks.BetweenBB(kingSq, sniperSq) & occNoEP;
        if (blockers != 0 && (blockers & (blockers - 1)) == 0)
        {
          pinned |= blockers;
        }
      }
      ctx.Pinned = pinned & ourOccupied;
    }


    /// <summary>
    /// Returns all squares attacked by the given diagonal and straight sliders
    /// for a specified occupancy.
    /// </summary>
    static BitBoard SliderAttackSet(BitBoard diagonal, BitBoard straight, BitBoard occupancy)
    {
      BitBoard attacks = 0;

      while (diagonal != 0)
      {
        attacks |= MGSliderAttacks.Bishop(BitOperations.TrailingZeroCount(diagonal), occupancy);
        diagonal &= diagonal - 1;
      }

      while (straight != 0)
      {
        attacks |= MGSliderAttacks.Rook(BitOperations.TrailingZeroCount(straight), occupancy);
        straight &= straight - 1;
      }

      return attacks;
    }


    /// <summary>
    /// Returns the enemy attack map to be used when testing the squares crossed by a
    /// castling king. The castling rook is removed from the occupancy (it vacates its
    /// square as part of the move), matching the semantics of the king-walk test which
    /// this replaces.
    /// </summary>
    internal static BitBoard DangerMapForCastling(in MGGenContext ctx, BitBoard rookSquare)
    {
      BitBoard occupancy = ctx.OccupiedNoEP & ~ctx.KingBB & ~rookSquare;
      return ctx.DangerNonSliders | SliderAttackSet(ctx.EnemyDiagonal, ctx.EnemyStraight, occupancy);
    }

    #endregion

    #region Entry points

    public static void GenerateMoves(in MGPosition P, MGMoveList moves)
    {
      Debug.Assert((~(P.A | P.B | P.C) & P.D) == 0); // Should not be any "black" empty squares

      if (P.BlackToMove)
      {
        DoGenBlackMoves(in P, moves, MoveGenMode.AllMoves);
      }
      else
      {
        DoGenWhiteMoves(in P, moves, MoveGenMode.AllMoves);
      }
    }


    [ThreadStatic]
    static MGMoveList movesTempForGeneratedMoves;

    /// <summary>
    /// Generates all legal moves for the given position and returns a new MGMoveList sized exactly to the number of moves.
    /// </summary>
    /// <param name="P"></param>
    /// <returns></returns>
    public static MGMoveList GeneratedMoves(in MGPosition P)
    {
      MGMoveList scratch = GeneratedMovesIntoThreadStaticScratch(in P);

      // Create a new MGMoveList sized exactly to the number of moves and copy
      int numMoves = scratch.NumMovesUsed;
      MGMoveList result = new MGMoveList(numMoves);
      result.NumMovesUsed = numMoves;
      scratch.MovesArray.AsSpan(0, numMoves).CopyTo(result.MovesArray);

      return result;
    }


    /// <summary>
    /// Generates all legal moves for the given position into a reusable thread-local scratch list
    /// (no allocation after warmup) and returns it.
    /// The returned list is invalidated by the next call on the same thread; callers which
    /// retain the list beyond that window must make an exactly-sized copy themselves.
    /// </summary>
    /// <param name="P"></param>
    /// <returns></returns>
    public static MGMoveList GeneratedMovesIntoThreadStaticScratch(in MGPosition P)
    {
      movesTempForGeneratedMoves ??= new MGMoveList();
      movesTempForGeneratedMoves.NumMovesUsed = 0;

      GenerateMoves(in P, movesTempForGeneratedMoves);

      return movesTempForGeneratedMoves;
    }


    [ThreadStatic]
    static MGMoveList movesTemp;

    public enum MoveGenMode { AllMoves, AtLeastOneMoveIfAnyExists };


    public static bool AtLeastOneLegalMoveExists(in MGPosition P)
    {
      if (movesTemp == null) movesTemp = new MGMoveList();
      movesTemp.NumMovesUsed = 0;

      if (P.BlackToMove)
        DoGenBlackMoves(in P, movesTemp, MoveGenMode.AtLeastOneMoveIfAnyExists);
      else
        DoGenWhiteMoves(in P, movesTemp, MoveGenMode.AtLeastOneMoveIfAnyExists);

      return movesTemp.NumMovesUsed > 0;
    }

    #endregion

    #region White move generation

    static void AddWhiteKnightMoves(in MGPosition P, in MGGenContext ctx, MGMoveList moves, byte q, BitBoard whiteFree)
    {
      BitBoard targets = MGSliderAttacks.KnightAttacks[q] & whiteFree;
      while (targets != 0)
      {
        int to = BitOperations.TrailingZeroCount(targets);
        targets &= targets - 1;
        DoAddWhiteMoveToListIfLegal(in P, in ctx, moves, q, (byte)to, 1UL << to, MGPositionConstants.WKNIGHT);
      }
    }


    static void DoGenWhiteMoves(in MGPosition P, MGMoveList moves, MoveGenMode mode)
    {
      Debug.Assert(moves.NumMovesUsed == 0);

      BitBoard occupied = P.A | P.B | P.C;                // all squares occupied by something
      BitBoard pABCTemp = (P.A & P.B & ~P.C);
      BitBoard whiteOccupied = (occupied & ~P.D) & ~pABCTemp; // all squares occupied by W, excluding EP Squares
      BitBoard blackOccupied = occupied & P.D;              // all squares occupied by B, including Black EP Squares
      BitBoard whiteFree;       // all squares where W is free to move
      whiteFree = pABCTemp  // any EP square
        | ~(occupied)       // any vacant square
        | (~P.A & P.D)      // Black Bishop, Rook or Queen
        | (~P.B & P.D);     // Black Pawn or Knight

      Debug.Assert(whiteOccupied != 0);

      BuildContext(in P, false, out MGGenContext ctx);

      BitBoard square;
      BitBoard currentSquare;
      BitBoard A, B, C;
      MGMove M = new MGMove(); // Dummy Move object used for setting flags.

      BitBoard occupiedYetToBeProcessed = whiteOccupied;
      byte q;
      while (occupiedYetToBeProcessed != 0)
      {
        // For efficiency, exit immediately if we are in "at least one move" mode and we have seen one or more moves
        if (mode == MoveGenMode.AtLeastOneMoveIfAnyExists && moves.NumMovesUsed > 0) return;

        q = (byte)BitOperations.TrailingZeroCount(occupiedYetToBeProcessed);
        currentSquare = 1UL << (int)q;
        occupiedYetToBeProcessed ^= currentSquare;

        A = P.A & currentSquare;
        B = P.B & currentSquare;
        C = P.C & currentSquare;

        if (A != 0)
        {
          if (B == 0)
          {
            if (C == 0)
            {
              // single move forward
              square = MGPositionConstants.MoveUp[q] & whiteFree & ~blackOccupied /* pawns can't capture in forward moves */;
              if ((square & MGPositionConstants.RANK8) != 0)
                AddWhitePromotionsToListIfLegal(in P, in ctx, moves, q, square, MGPositionConstants.WPAWN);
              else
              {
                // Ordinary Pawn Advance
                AddWhiteMoveToListIfLegal(in P, in ctx, moves, q, square, MGPositionConstants.WPAWN);
                /* double move forward (only available from 2nd Rank */
                if ((currentSquare & MGPositionConstants.RANK2) != 0)
                {
                  square = MGMoveGenFillFunctions.MoveUpSingleOccluded(square, whiteFree) & ~blackOccupied;
                  M.Flags = 0;
                  M.DoublePawnMove = true; // this flag will cause ChessPosition::performMove() to set an ep square in the position
                  AddWhiteMoveToListIfLegal(in P, in ctx, moves, q, square, MGPositionConstants.WPAWN, M.Flags);
                }
              }
              // generate Pawn Captures:
              square = MGPositionConstants.MoveUpLeft[q] & whiteFree & blackOccupied;
              if ((square & MGPositionConstants.RANK8) != 0)
                AddWhitePromotionsToListIfLegal(in P, in ctx, moves, q, square, MGPositionConstants.WPAWN);
              else
              {
                // Ordinary Pawn Capture to Left
                AddWhiteMoveToListIfLegal(in P, in ctx, moves, q, square, MGPositionConstants.WPAWN);
              }
              square = MGPositionConstants.MoveUpRight[q] & whiteFree & blackOccupied;
              if ((square & MGPositionConstants.RANK8) != 0)
                AddWhitePromotionsToListIfLegal(in P, in ctx, moves, q, square, MGPositionConstants.WPAWN);
              else
              {
                // Ordinary Pawn Capture to right
                AddWhiteMoveToListIfLegal(in P, in ctx, moves, q, square, MGPositionConstants.WPAWN);
              }
              continue;
            }
            else
            {
              AddWhiteKnightMoves(in P, in ctx, moves, q, whiteFree);
              continue;
            }
          } // Ends if (B == 0)

          else /* B != 0 */
          {
            if (C != 0)
            {
              BitBoard targets = MGSliderAttacks.KingAttacks[q] & whiteFree;
              while (targets != 0)
              {
                int to = BitOperations.TrailingZeroCount(targets);
                targets &= targets - 1;
                DoAddWhiteMoveToListIfLegal(in P, in ctx, moves, q, (byte)to, 1UL << to, MGPositionConstants.WKING);
              }

              ulong rookSquare;
              // Conditionally generate O-O move:
              if (P.WhiteCanCastle && QBBoperations.CanWhiteKingReachShortRook(in P, in ctx, out rookSquare))
              {
                // OK to Castle
                M.Flags = 0;
                M.CastleShort = true;
                AddWhiteMoveToListIfLegal(in P, in ctx, moves, q, rookSquare, MGPositionConstants.WKING, M.Flags);
              }

              // Conditionally generate O-O-O move:
              if (P.WhiteCanCastleLong && QBBoperations.CanWhiteKingReachLongRook(in P, in ctx, out rookSquare))
              {
                // Ok to Castle Long
                M.Flags = 0;
                M.CastleLong = true;
                AddWhiteMoveToListIfLegal(in P, in ctx, moves, q, rookSquare, MGPositionConstants.WKING, M.Flags);
              }

              continue;
            } // ENDS if (C != 0)
            else
            {
              // en passant square - no action to be taken, but continue loop
              continue;
            }
          } // Ends else
        } // ENDS if (A !=0)

        {
          // Bishop, rook or queen (A is zero here). A single table lookup per ray
          // family yields the whole attack set, which is then serialized.
          ulong piece;
          BitBoard targets;
          if (B != 0)
          {
            targets = MGSliderAttacks.Bishop(q, ctx.OccupiedNoEP);
            if (C != 0)
            {
              targets |= MGSliderAttacks.Rook(q, ctx.OccupiedNoEP);
              piece = MGPositionConstants.WQUEEN;
            }
            else
            {
              piece = MGPositionConstants.WBISHOP;
            }
          }
          else
          {
            targets = MGSliderAttacks.Rook(q, ctx.OccupiedNoEP);
            piece = MGPositionConstants.WROOK;
          }

          targets &= whiteFree;
          while (targets != 0)
          {
            int to = BitOperations.TrailingZeroCount(targets);
            targets &= targets - 1;
            DoAddWhiteMoveToListIfLegal(in P, in ctx, moves, q, (byte)to, 1UL << to, piece);
          }
        }
      }

      // Create 'no more moves' move to mark end of list:
      moves.MovesArray[moves.NumMovesUsed].FromSquareIndex = 0;
      moves.MovesArray[moves.NumMovesUsed].ToSquareIndex = 0;
      moves.MovesArray[moves.NumMovesUsed].Piece = 0;
      moves.MovesArray[moves.NumMovesUsed].NoMoreMoves = true;
    }


    static void AddWhiteMoveToListIfLegal(in MGPosition P, in MGGenContext ctx, MGMoveList moves,
                                          byte fromsquare, BitBoard to, ulong piece, MGMove.MGChessMoveFlags flags = 0)
    {
      if (to != 0)
      {
        DoAddWhiteMoveToListIfLegal(in P, in ctx, moves, fromsquare, MGMoveGenFillFunctions.GetSquareIndex(to), to, piece, flags);
      }
    }


    static void DoAddWhiteMoveToListIfLegal(in MGPosition P, in MGGenContext ctx, MGMoveList moves,
                                            byte fromsquare, byte tosquare, BitBoard to, ulong piece,
                                            MGMove.MGChessMoveFlags flags = 0)
    {
      moves.InsureMoveArrayHasRoom(1);

      ref MGMove thisMove = ref moves.MovesArray[moves.NumMovesUsed];

      thisMove.FromSquareIndex = fromsquare;
      thisMove.ToSquareIndex = tosquare;
      thisMove.Flags = flags;
      thisMove.Piece = (MGPositionConstants.MCChessPositionPieceEnum)piece;

      bool legal;

      if (thisMove.CastleShort)
      {
        // The destination square of a castling move encodes the castling rook.
        // Toggling king and rook independently (rather than or-ing the two square
        // pairs together) keeps the trial board correct when either piece already
        // stands on the other's destination square, as is possible in Chess960.
        BitBoard kingToggle = (1UL << fromsquare) ^ G1;
        BitBoard rookToggle = to ^ F1;
        legal = !IsWhiteInCheck(P.A ^ kingToggle, P.B ^ kingToggle, P.C ^ kingToggle ^ rookToggle, P.D);
      }

      else if (thisMove.CastleLong)
      {
        BitBoard kingToggle = (1UL << fromsquare) ^ C1;
        BitBoard rookToggle = to ^ D1;
        legal = !IsWhiteInCheck(P.A ^ kingToggle, P.B ^ kingToggle, P.C ^ kingToggle ^ rookToggle, P.D);
      }

      else
      {
        // Test for capture:
        if ((to & P.D) != 0)
        {
          BitBoard PAB = (P.A & P.B); // Bitboard containing EnPassants and kings:
          if ((to & ~PAB) != 0) // Only considered a capture if dest is not an enpassant or king.
            thisMove.Capture = true;

          else if ((piece == MGPositionConstants.WPAWN) && (to & PAB & ~P.C) != 0)
            thisMove.EnPassantCapture = true;
        }

        if (thisMove.EnPassantCapture)
        {
          // En passant removes a piece which is not on the destination square, so the
          // pin/check masks do not describe it. This is rare enough to test explicitly.
          BitBoard O = ~((1UL << fromsquare) | to);
          BitBoard captured = to >> 8;
          BitBoard clear = ~captured & O;
          BitBoard QA = (P.A & clear) | to;   // WPAWN is A only
          BitBoard QB = P.B & clear;
          BitBoard QC = P.C & clear;
          BitBoard QD = P.D & clear;
          legal = !IsWhiteInCheck(QA, QB, QC, QD);
        }
        else if (piece == MGPositionConstants.WKING)
        {
          legal = (ctx.DangerMap & to) == 0;
        }
        else
        {
          legal = ctx.OrdinaryMoveIsLegal(fromsquare, to);
        }
      }

      if (legal)
      {
        moves.NumMovesUsed++;     // Advancing the pointer means that the
                                  // move is now added to the list.
                                  // (pointer is ready for next move)
        moves.MovesArray[moves.NumMovesUsed].Flags = 0;
      }
      else
        thisMove.IllegalMove = true;
    }


    static void AddWhitePromotionsToListIfLegal(in MGPosition P, in MGGenContext ctx, MGMoveList moves,
                                                byte fromsquare, BitBoard to, ulong piece, MGMove.MGChessMoveFlags flags = 0)
    {
      if (to != 0)
      {
        moves.InsureMoveArrayHasRoom(4);

        ref MGMove thisMove = ref moves.MovesArray[moves.NumMovesUsed];

        thisMove.FromSquareIndex = fromsquare;
        thisMove.ToSquareIndex = MGMoveGenFillFunctions.GetSquareIndex(to);
        thisMove.Flags = flags;
        thisMove.Piece = (MGPositionConstants.MCChessPositionPieceEnum)piece;

        // Test for capture:
        BitBoard PAB = (P.A & P.B); // Bitboard containing EnPassants and kings:
        thisMove.Capture = (to & P.D & ~PAB) != 0;

        // The promoted-to piece cannot affect whether our own king is left in check,
        // so the ordinary from/to legality test covers all four promotions.
        if (ctx.OrdinaryMoveIsLegal(fromsquare, to))
        {
          // make an additional 3 copies (there are four promotions)
          moves.MovesArray[moves.NumMovesUsed + 1] = thisMove;
          moves.MovesArray[moves.NumMovesUsed + 2] = thisMove;
          moves.MovesArray[moves.NumMovesUsed + 3] = thisMove;

          // set Promotion flags accordingly:
          moves.MovesArray[moves.NumMovesUsed].PromoteKnight = true;
          moves.MovesArray[moves.NumMovesUsed + 1].PromoteBishop = true;
          moves.MovesArray[moves.NumMovesUsed + 2].PromoteRook = true;
          moves.MovesArray[moves.NumMovesUsed + 3].PromoteQueen = true;

          moves.MovesArray[moves.NumMovesUsed + 4].Flags = 0;
          moves.NumMovesUsed += 4;
        }
        else
          moves.MovesArray[moves.NumMovesUsed].IllegalMove = true;
      }
    }

    #endregion

    #region Black move generation

    static void AddBlackKnightMoves(in MGPosition P, in MGGenContext ctx, MGMoveList moves, byte q, BitBoard blackFree)
    {
      BitBoard targets = MGSliderAttacks.KnightAttacks[q] & blackFree;
      while (targets != 0)
      {
        int to = BitOperations.TrailingZeroCount(targets);
        targets &= targets - 1;
        DoAddBlackMoveToListIfLegal(in P, in ctx, moves, q, (byte)to, 1UL << to, MGPositionConstants.BKNIGHT);
      }
    }


    static void DoGenBlackMoves(in MGPosition P, MGMoveList moves, MoveGenMode mode)
    {
      Debug.Assert(moves.NumMovesUsed == 0);

      BitBoard occupied = P.A | P.B | P.C;                // all squares occupied by something
      BitBoard pABCTemp = (P.A & P.B & ~P.C);
      BitBoard blackOccupied = P.D & ~pABCTemp;         // all squares occupied by B, excluding EP Squares
      BitBoard whiteOccupied = (occupied & ~P.D);             // all squares occupied by W, including white EP Squares
      BitBoard blackFree;       // all squares where B is free to move
      blackFree = pABCTemp  // any EP square
        | ~(occupied)       // any vacant square
        | (~P.A & ~P.D)       // White Bishop, Rook or Queen
        | (~P.B & ~P.D);      // White Pawn or Knight

      Debug.Assert(blackOccupied != 0);

      BuildContext(in P, true, out MGGenContext ctx);

      BitBoard square;
      BitBoard currentSquare;
      BitBoard A, B, C;
      MGMove M = new MGMove(); // Dummy Move object used for setting flags.

      BitBoard occupiedYetToBeProcessed = blackOccupied;
      byte q;
      while (occupiedYetToBeProcessed != 0)
      {
        // For efficiency, exit immediately if we are in "at least one move" mode and we have seen one or more moves
        if (mode == MoveGenMode.AtLeastOneMoveIfAnyExists && moves.NumMovesUsed > 0) return;

        q = (byte)BitOperations.TrailingZeroCount(occupiedYetToBeProcessed);
        currentSquare = 1UL << (int)q;
        occupiedYetToBeProcessed ^= currentSquare;

        A = P.A & currentSquare;
        B = P.B & currentSquare;
        C = P.C & currentSquare;

        if (A != 0)
        {
          if (B == 0)
          {
            if (C == 0)
            {
              // single move forward
              square = MGPositionConstants.MoveDown[q] & blackFree & ~whiteOccupied /* pawns can't capture in forward moves */;
              if ((square & MGPositionConstants.RANK1) != 0)
                AddBlackPromotionsToListIfLegal(in P, in ctx, moves, q, square, MGPositionConstants.BPAWN);
              else
              {
                // Ordinary Pawn Advance
                AddBlackMoveToListIfLegal(in P, in ctx, moves, q, square, MGPositionConstants.BPAWN);
                /* double move forward (only available from 7th Rank */
                if ((currentSquare & MGPositionConstants.RANK7) != 0)
                {
                  square = MGMoveGenFillFunctions.MoveDownSingleOccluded(square, blackFree) & ~whiteOccupied;
                  M.Flags = 0;
                  M.DoublePawnMove = true; // this flag will cause ChessPosition::performMove() to set an ep square in the position
                  AddBlackMoveToListIfLegal(in P, in ctx, moves, q, square, MGPositionConstants.BPAWN, M.Flags);
                }
              }
              // generate Pawn Captures:
              square = MGPositionConstants.MoveDownLeft[q] & blackFree & whiteOccupied;
              if ((square & MGPositionConstants.RANK1) != 0)
                AddBlackPromotionsToListIfLegal(in P, in ctx, moves, q, square, MGPositionConstants.BPAWN);
              else
              {
                // Ordinary Pawn Capture to Left
                AddBlackMoveToListIfLegal(in P, in ctx, moves, q, square, MGPositionConstants.BPAWN);
              }
              square = MGPositionConstants.MoveDownRight[q] & blackFree & whiteOccupied;
              if ((square & MGPositionConstants.RANK1) != 0)
                AddBlackPromotionsToListIfLegal(in P, in ctx, moves, q, square, MGPositionConstants.BPAWN);
              else
              {
                // Ordinary Pawn Capture to right
                AddBlackMoveToListIfLegal(in P, in ctx, moves, q, square, MGPositionConstants.BPAWN);
              }
              continue;
            }
            else
            {
              AddBlackKnightMoves(in P, in ctx, moves, q, blackFree);
              continue;
            }
          } // ENDS if (B ==0)

          else /* B != 0 */
          {
            if (C != 0)
            {
              BitBoard targets = MGSliderAttacks.KingAttacks[q] & blackFree;
              while (targets != 0)
              {
                int to = BitOperations.TrailingZeroCount(targets);
                targets &= targets - 1;
                DoAddBlackMoveToListIfLegal(in P, in ctx, moves, q, (byte)to, 1UL << to, MGPositionConstants.BKING);
              }

              ulong rookSquare;
              // Conditionally generate O-O move:
              if (P.BlackCanCastle && QBBoperations.CanBlackKingReachShortRook(in P, in ctx, out rookSquare))
              {
                // OK to Castle
                M.Flags = 0;
                M.CastleShort = true;
                AddBlackMoveToListIfLegal(in P, in ctx, moves, q, rookSquare, MGPositionConstants.BKING, M.Flags);
              }

              // Conditionally generate O-O-O move:
              if (P.BlackCanCastleLong && QBBoperations.CanBlackKingReachLongRook(in P, in ctx, out rookSquare))
              {
                // OK to castle Long
                M.Flags = 0;
                M.CastleLong = true;
                AddBlackMoveToListIfLegal(in P, in ctx, moves, q, rookSquare, MGPositionConstants.BKING, M.Flags);
              }

              continue;
            } // ENDS if (C != 0)
            else
            {
              // en passant square - no action to be taken, but continue loop
              continue;
            }
          } // Ends else
        } // ENDS if (A !=0)

        {
          // Bishop, rook or queen (A is zero here).
          ulong piece;
          BitBoard targets;
          if (B != 0)
          {
            targets = MGSliderAttacks.Bishop(q, ctx.OccupiedNoEP);
            if (C != 0)
            {
              targets |= MGSliderAttacks.Rook(q, ctx.OccupiedNoEP);
              piece = MGPositionConstants.BQUEEN;
            }
            else
            {
              piece = MGPositionConstants.BBISHOP;
            }
          }
          else
          {
            targets = MGSliderAttacks.Rook(q, ctx.OccupiedNoEP);
            piece = MGPositionConstants.BROOK;
          }

          targets &= blackFree;
          while (targets != 0)
          {
            int to = BitOperations.TrailingZeroCount(targets);
            targets &= targets - 1;
            DoAddBlackMoveToListIfLegal(in P, in ctx, moves, q, (byte)to, 1UL << to, piece);
          }
        }
      }

      // Create 'no more moves' move to mark end of list:
      moves.MovesArray[moves.NumMovesUsed].FromSquareIndex = 0;
      moves.MovesArray[moves.NumMovesUsed].ToSquareIndex = 0;
      moves.MovesArray[moves.NumMovesUsed].Piece = 0;
      moves.MovesArray[moves.NumMovesUsed].NoMoreMoves = true;
    }


    static void AddBlackMoveToListIfLegal(in MGPosition P, in MGGenContext ctx, MGMoveList moves,
                                          byte fromsquare, BitBoard to, ulong piece, MGMove.MGChessMoveFlags flags = 0)
    {
      if (to != 0)
      {
        DoAddBlackMoveToListIfLegal(in P, in ctx, moves, fromsquare, MGMoveGenFillFunctions.GetSquareIndex(to), to, piece, flags);
      }
    }


    static void DoAddBlackMoveToListIfLegal(in MGPosition P, in MGGenContext ctx, MGMoveList moves,
                                            byte fromsquare, byte tosquare, BitBoard to, ulong piece,
                                            MGMove.MGChessMoveFlags flags = 0)
    {
      moves.InsureMoveArrayHasRoom(1);

      ref MGMove thisMove = ref moves.MovesArray[moves.NumMovesUsed];

      thisMove.FromSquareIndex = fromsquare;
      thisMove.ToSquareIndex = tosquare;
      thisMove.Flags = flags;
      thisMove.BlackToMove = true;
      thisMove.Piece = (MGPositionConstants.MCChessPositionPieceEnum)piece;

      bool legal;

      if (thisMove.CastleShort)
      {
        BitBoard kingToggle = (1UL << fromsquare) ^ G8;
        BitBoard rookToggle = to ^ F8;
        BitBoard both = kingToggle ^ rookToggle;
        legal = !IsBlackInCheck(P.A ^ kingToggle, P.B ^ kingToggle, P.C ^ both, P.D ^ both);
      }

      else if (thisMove.CastleLong)
      {
        BitBoard kingToggle = (1UL << fromsquare) ^ C8;
        BitBoard rookToggle = to ^ D8;
        BitBoard both = kingToggle ^ rookToggle;
        legal = !IsBlackInCheck(P.A ^ kingToggle, P.B ^ kingToggle, P.C ^ both, P.D ^ both);
      }

      else
      {
        // Test for capture:
        BitBoard whiteOccupied = (P.A | P.B | P.C) & ~P.D;
        if ((to & whiteOccupied) != 0)
        {
          BitBoard PAB = (P.A & P.B); // Bitboard containing EnPassants and kings
          if ((to & ~PAB) != 0) // Only considered a capture if dest is not an enpassant or king.
            thisMove.Capture = true;

          else if ((piece == MGPositionConstants.BPAWN) && (to & PAB & ~P.C) != 0)
            thisMove.EnPassantCapture = true;
        }

        if (thisMove.EnPassantCapture)
        {
          BitBoard O = ~((1UL << fromsquare) | to);
          BitBoard captured = to << 8;
          BitBoard clear = ~captured & O;
          BitBoard QA = (P.A & clear) | to;   // BPAWN is A and D
          BitBoard QB = P.B & clear;
          BitBoard QC = P.C & clear;
          BitBoard QD = (P.D & clear) | to;
          legal = !IsBlackInCheck(QA, QB, QC, QD);
        }
        else if (piece == MGPositionConstants.BKING)
        {
          legal = (ctx.DangerMap & to) == 0;
        }
        else
        {
          legal = ctx.OrdinaryMoveIsLegal(fromsquare, to);
        }
      }

      if (legal)
      {
        moves.NumMovesUsed++;     // Advancing the pointer means that the
                                  // move is now added to the list.
                                  // (pointer is ready for next move)
        moves.MovesArray[moves.NumMovesUsed].Flags = 0;
      }
      else
        thisMove.IllegalMove = true;
    }


    static void AddBlackPromotionsToListIfLegal(in MGPosition P, in MGGenContext ctx, MGMoveList moves,
                                                byte fromsquare, BitBoard to, ulong piece, MGMove.MGChessMoveFlags flags = 0)
    {
      if (to != 0)
      {
        moves.InsureMoveArrayHasRoom(4);

        ref MGMove thisMove = ref moves.MovesArray[moves.NumMovesUsed];

        thisMove.FromSquareIndex = fromsquare;
        thisMove.ToSquareIndex = MGMoveGenFillFunctions.GetSquareIndex(to);
        thisMove.Flags = flags;
        thisMove.BlackToMove = true;
        thisMove.Piece = (MGPositionConstants.MCChessPositionPieceEnum)piece;

        // Test for capture:
        BitBoard PAB = (P.A & P.B); // Bitboard containing EnPassants and kings
        BitBoard whiteOccupied = (P.A | P.B | P.C) & ~P.D;
        thisMove.Capture = (to & whiteOccupied & ~PAB) != 0;

        if (ctx.OrdinaryMoveIsLegal(fromsquare, to))
        {
          // make an additional 3 copies for underpromotions
          moves.MovesArray[moves.NumMovesUsed + 1] = thisMove;
          moves.MovesArray[moves.NumMovesUsed + 2] = thisMove;
          moves.MovesArray[moves.NumMovesUsed + 3] = thisMove;

          // set Promotion flags accordingly:
          moves.MovesArray[moves.NumMovesUsed].PromoteQueen = true;
          moves.MovesArray[moves.NumMovesUsed + 1].PromoteRook = true;
          moves.MovesArray[moves.NumMovesUsed + 2].PromoteBishop = true;
          moves.MovesArray[moves.NumMovesUsed + 3].PromoteKnight = true;
          moves.MovesArray[moves.NumMovesUsed + 4].Flags = 0;
          moves.NumMovesUsed += 4;
        }
        else
          moves.MovesArray[moves.NumMovesUsed].IllegalMove = true;
      }
    }

    #endregion

    #region Castling square constants

    // Destination squares of the king and rook when castling
    // (recall bit 0 is h1, so g1 is bit 1 and c1 is bit 5).
    const BitBoard G1 = 1UL << 1;
    const BitBoard F1 = 1UL << 2;
    const BitBoard C1 = 1UL << 5;
    const BitBoard D1 = 1UL << 4;
    const BitBoard G8 = 1UL << 57;
    const BitBoard F8 = 1UL << 58;
    const BitBoard C8 = 1UL << 61;
    const BitBoard D8 = 1UL << 60;

    #endregion

    #region Attack sets and check detection

    /// <summary>
    /// Returns the bitboard of all squares attacked by one side
    /// (public wrapper over the internal attack generators).
    /// </summary>
    public static BitBoard AttackedSquares(in MGPosition pos, bool byBlack)
      => byBlack ? GenBlackAttacks(pos) : GenWhiteAttacks(pos);


    static BitBoard GenWhiteAttacks(MGPosition Z)
    {
      BitBoard Occupied = Z.A | Z.B | Z.C;
      BitBoard Empty = (Z.A & Z.B & ~Z.C) | // All EP squares, regardless of colour
        ~Occupied;              // All Unoccupied squares

      BitBoard PotentialCapturesForWhite = Occupied & Z.D; // Black Pieces (including Kings)

      BitBoard A = Z.A & ~Z.D;        // White A-Plane
      BitBoard B = Z.B & ~Z.D;        // White B-Plane
      BitBoard C = Z.C & ~Z.D;        // White C-Plane

      BitBoard S = C & ~A;        // Straight-moving Pieces
      BitBoard D = B & ~A;        // Diagonal-moving Pieces
      BitBoard K = A & B & C;       // King
      BitBoard P = A & ~B & ~C;     // Pawns
      BitBoard N = A & ~B & C;      // Knights

      BitBoard allowed = Empty | PotentialCapturesForWhite;
      BitBoard attacks = SliderAttackSet(D, S, Occupied & ~Empty) & allowed;
      attacks |= MGMoveGenFillFunctions.FillKingAttacksOccluded(K, allowed);
      attacks |= MGMoveGenFillFunctions.FillKnightAttacksOccluded(N, allowed);
      attacks |= MGMoveGenFillFunctions.MoveUpLeftRightSingle(P) & allowed;

      return attacks;
    }


    static BitBoard GenBlackAttacks(MGPosition Z)
    {
      BitBoard Occupied = Z.A | Z.B | Z.C;
      BitBoard Empty = (Z.A & Z.B & ~Z.C) | // All EP squares, regardless of colour
                      ~Occupied;            // All Unoccupied squares

      BitBoard PotentialCapturesForBlack = Occupied & ~Z.D; // White Pieces (including Kings)

      BitBoard A = Z.A & Z.D;       // Black A-Plane
      BitBoard B = Z.B & Z.D;       // Black B-Plane
      BitBoard C = Z.C & Z.D;       // Black C-Plane

      BitBoard S = C & ~A;        // Straight-moving Pieces
      BitBoard D = B & ~A;        // Diagonal-moving Pieces
      BitBoard K = A & B & C;       // King
      BitBoard P = A & ~B & ~C;     // Pawns
      BitBoard N = A & ~B & C;      // Knights

      BitBoard allowed = Empty | PotentialCapturesForBlack;
      BitBoard attacks = SliderAttackSet(D, S, Occupied & ~Empty) & allowed;
      attacks |= MGMoveGenFillFunctions.FillKingAttacksOccluded(K, allowed);
      attacks |= MGMoveGenFillFunctions.FillKnightAttacksOccluded(N, allowed);
      attacks |= MGMoveGenFillFunctions.MoveDownLeftRightSingle(P) & allowed;

      return attacks;
    }


    internal static bool IsWhiteInCheck(BitBoard ZA, BitBoard ZB, BitBoard ZC, BitBoard ZD)
    {
      BitBoard ZAandZB = ZA & ZB;
      BitBoard whiteKing = ZAandZB & ZC & ~ZD;
      if (whiteKing == 0)
      {
        return false;
      }

      int kingSq = BitOperations.TrailingZeroCount(whiteKing);

      // En passant markers stand on physically empty squares and do not block sliders.
      BitBoard occupied = (ZA | ZB | ZC) & ~(ZAandZB & ~ZC);

      BitBoard A = ZA & ZD;       // Black A-Plane
      BitBoard B = ZB & ZD;       // Black B-Plane
      BitBoard C = ZC & ZD;       // Black C-Plane
      BitBoard AnotB = A & ~B;

      return ((MGMoveGenFillFunctions.MoveUpLeftRightSingle(whiteKing) & AnotB & ~C) != 0)  // Pawns
          || ((MGSliderAttacks.KnightAttacks[kingSq] & AnotB & C) != 0)                     // Knights
          || ((MGSliderAttacks.KingAttacks[kingSq] & A & B & C) != 0)                       // King
          || ((MGSliderAttacks.Bishop(kingSq, occupied) & B & ~A) != 0)                     // Bishops, Queens
          || ((MGSliderAttacks.Rook(kingSq, occupied) & C & ~A) != 0);                      // Rooks, Queens
    }


    internal static bool IsBlackInCheck(BitBoard ZA, BitBoard ZB, BitBoard ZC, BitBoard ZD)
    {
      BitBoard ZAandZB = ZA & ZB;
      BitBoard blackKing = ZAandZB & ZC & ZD;
      if (blackKing == 0)
      {
        return false;
      }

      int kingSq = BitOperations.TrailingZeroCount(blackKing);

      BitBoard occupied = (ZA | ZB | ZC) & ~(ZAandZB & ~ZC);

      BitBoard A = ZA & ~ZD;      // White A-Plane
      BitBoard B = ZB & ~ZD;      // White B-Plane
      BitBoard C = ZC & ~ZD;      // White C-Plane
      BitBoard AnotB = A & ~B;

      return ((MGMoveGenFillFunctions.MoveDownLeftRightSingle(blackKing) & AnotB & ~C) != 0) // Pawns
          || ((MGSliderAttacks.KnightAttacks[kingSq] & AnotB & C) != 0)                      // Knights
          || ((MGSliderAttacks.KingAttacks[kingSq] & A & B & C) != 0)                        // King
          || ((MGSliderAttacks.Bishop(kingSq, occupied) & B & ~A) != 0)                      // Bishops, Queens
          || ((MGSliderAttacks.Rook(kingSq, occupied) & C & ~A) != 0);                       // Rooks, Queens
    }


    public static bool IsInCheck(in MGPosition P, bool bIsBlack)
    {
      return bIsBlack ? IsBlackInCheck(P.A, P.B, P.C, P.D)
                      : IsWhiteInCheck(P.A, P.B, P.C, P.D);
    }

    #endregion
  }
}
