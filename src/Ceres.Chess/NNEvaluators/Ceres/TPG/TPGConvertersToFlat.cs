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
using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NNEvaluators;
//using CeresTrain.NNEvaluators;
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using Ceres.Chess.PositionDataInfo;

#endregion 

namespace Ceres.Chess.NNEvaluators.Ceres.TPG
{
  /// <summary>
  /// Static helper methods to convert TPGRecord to flat square format.
  /// </summary>
  public static class TPGConvertersToFlat
  {
    /// <summary>
    /// Converts a batch of TPGRecord[] into TPG flat square values.
    /// </summary>
    /// <param name="records"></param>
    /// <param name="flatValues"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public static int ConvertToFlatTPGFromTPG(NNEvaluatorOptions options, object records, Span<byte> flatValues)
    {
      NNEvaluatorOptionsCeres optionsCeres = (NNEvaluatorOptionsCeres)options;

      // TODO: Requiring the converter to take a materialized array could be inefficient, can we use Memory instead?
      TPGRecord[] tpgRecords = records as TPGRecord[];
      if (tpgRecords == null)
      {
        throw new NotImplementedException("Expected input to be TPGRecord[]");
      }

      byte[] squareBytesAll = new byte[tpgRecords.Length * Marshal.SizeOf<TPGSquareRecord>() * 64];

      for (int i = 0; i < tpgRecords.Length; i++)
      {
        int offsetSquares = i * 64 * TPGRecord.BYTES_PER_SQUARE_RECORD;

        // Extract as bytes.
        tpgRecords[i].CopySquares(squareBytesAll, offsetSquares);
      }

      // N.B. Scaling (by 100) already done in ONXNRuntimeExecutor (but probably doesn't belong there?).
      // TODO: This could be made more efficient, fold into the above loop.
      for (int i = 0; i < squareBytesAll.Length; i++)
      {
        flatValues[i] = squareBytesAll[i];
      }

      return squareBytesAll.Length;
    }


    /// <summary>
    /// Copies sourceBytes into targetFloats, also dividing by divisor.
    /// </summary>
    /// <param name="sourceBytes"></param>
    /// <param name="targetHalves"></param>
    static void CopyAndDivide(Memory<byte> sourceBytes, Memory<Half> targetHalves, float divisor)
    {
      CopyAndDivideSIMD(sourceBytes, targetHalves, divisor);
#if NOT
      // Disabled. Incompatible with the MemoryHandle.Pin used in CopyAndDivideSIMD that objects to re-pinning.
      // Also the parallelism may not be very helpful or needed given that the common path now uses
      // byte inputs and completely avoids this code path.
      const int CHUNK_SIZE = 2 * 1024 * 128;
      if (sourceBytes.Length >= CHUNK_SIZE * 2)
      {
        Parallel.For(0, sourceBytes.Length / CHUNK_SIZE + 1,
                     new ParallelOptions()
                     {
                       MaxDegreeOfParallelism = 5 // limit parallelism because already threaded if multiple GPUs
                     },
          (chunkIndex) =>
          {
            int startIndex = chunkIndex * CHUNK_SIZE;
            int numThisBlock = Math.Min(CHUNK_SIZE, sourceBytes.Length - startIndex);

            if (numThisBlock > 0)
            {
              Memory<byte> sourceBytesThisBlock = sourceBytes.Slice(startIndex, numThisBlock);
              Memory<Half> targetHalvesThisBlock = targetHalves.Slice(startIndex, numThisBlock);
              CopyAndDivideSIMD(sourceBytesThisBlock, targetHalvesThisBlock, divisor);
            }
          });
      }
#endif
    }


    /// <summary>
    /// Number of elements converted per tile by CopyAndDivideSIMD.
    ///
    /// The float intermediate is 4x the size of the byte source, so converting a whole batch in one
    /// pass would push tens of megabytes through memory three times (widen, scale, narrow). Working
    /// in tiles keeps the intermediate resident in L1/L2 so only the source and destination touch
    /// memory. Measured on x64 (AVX-512 capable) over 0.5 - 9 MB inputs, a 4096-element tile was
    /// consistently the fastest of 4096 / 8192 / 16384 / 32768 / 65536 and was ~1.5x faster than the
    /// former hand-written AVX2 implementation; the untiled form was ~4x SLOWER than that AVX2 code.
    /// </summary>
    const int COPY_AND_DIVIDE_TILE = 4096;

    /// <summary>
    /// Scratch buffer holding one tile of widened float values (see CopyAndDivideSIMD).
    /// Thread static because conversion runs concurrently on multiple GPU worker threads.
    /// </summary>
    [ThreadStatic]
    static float[] copyAndDivideScratch;


    /// <summary>
    /// Copies sourceBytes into targetHalfs, also dividing by divisor.
    ///
    /// Implemented with TensorPrimitives so that it vectorizes on every architecture (in particular
    /// the narrowing to Half lowers to native FCVTN on ARM64, which has no F16C equivalent).
    /// This replaced a hand-written AVX2 implementation which it outperformed on x64.
    /// </summary>
    /// <param name="sourceBytes"></param>
    /// <param name="targetHalfs"></param>
    /// <param name="divisor"></param>
    internal static void CopyAndDivideSIMD(Memory<byte> sourceBytes, Memory<Half> targetHalfs, float divisor)
    {
      ReadOnlySpan<byte> source = sourceBytes.Span;
      Span<Half> target = targetHalfs.Span;

      if (target.Length < source.Length)
      {
        throw new ArgumentException($"Target length {target.Length} is less than source length {source.Length}");
      }

      // Scale by the reciprocal rather than dividing: measured ~25% faster.
      float reciprocal = 1.0f / divisor;

      float[] scratch = copyAndDivideScratch ??= new float[COPY_AND_DIVIDE_TILE];

      for (int offset = 0; offset < source.Length; offset += COPY_AND_DIVIDE_TILE)
      {
        int count = Math.Min(COPY_AND_DIVIDE_TILE, source.Length - offset);
        Span<float> tile = scratch.AsSpan(0, count);

        TensorPrimitives.ConvertChecked<byte, float>(source.Slice(offset, count), tile);
        TensorPrimitives.Multiply(tile, reciprocal, tile);
        TensorPrimitives.ConvertToHalf(tile, target.Slice(offset, count));
      }
    }


    [ThreadStatic] static byte[] squareValuesByteTemporary;

    /// <summary>
    /// Converts a IEncodedPositionBatchFlat of encoded positions into TPG flat square values.
    /// </summary>
    /// <param name="options"></param>
    /// <param name="batch"></param>
    /// <param name="includeHistory"></param>
    /// <param name="squareValues"></param>
    /// <exception cref="NotImplementedException"></exception>
    public static void ConvertToFlatTPG(NNEvaluatorOptions options,
                                        IEncodedPositionBatchFlat batch,
                                        bool includeHistory, Memory<byte> squareValuesByte, Memory<Half> squareValues, short[] legalMoveIndices)
    {
      TPGRecord tpgRecord = default;
      EncodedPositionBatchFlat ebf = batch as EncodedPositionBatchFlat;
      bool EMIT_PLY_SINCE = ebf?.LastMovePlies != null;

      // ConvertPositionsToRawSquareBytes reads history exclusively from batch.CompactHistories
      // (the canonical representation, derived from planes at the choke point by hook 1 for
      // plane-only producers). The former per-batch PositionsBuffer copy no longer exists.
      // TODO: someday handle since ply, does that need to be passed in from the search engine?

      byte[] moveBytesAll;

      NNEvaluatorOptionsCeres optionsCeres = options as NNEvaluatorOptionsCeres;

      // V3 TPG layout: WritePosPieces unconditionally writes 141 bytes per square
      // (137 base + 4 aux features baked in via PerSquareAttacks). The trained model can
      // consume either 137 (legacy, aux-blind) or 141 (V3). Width is auto-detected from
      // the caller's output buffer; the extra aux bytes are sliced off the per-square tail.
      int bytesPerSquareOut = TPGRecord.BYTES_PER_SQUARE_RECORD;  // 141 with V3=true, 137 with V3=false
      if (!squareValuesByte.IsEmpty)
      {
        int sizePerPos = squareValuesByte.Length / batch.NumPos;
        if (sizePerPos == 64 * 137)
        {
          bytesPerSquareOut = 137;  // legacy 137-channel model — slice off all 4 aux bytes
        }
        else if (sizePerPos == 64 * 141)
        {
          // V3 141-channel model — requires a V3 build: in a V2 build (USE_V2_TPG_RECORD,
          // BYTES_PER_SQUARE_RECORD == 137) WritePosPieces emits 137-byte squares with no
          // aux bytes, so a 141-stride output buffer would be filled misaligned — every
          // square's features shifted — producing silently garbage inference (policy KLD ~4,
          // near-zero accuracy) rather than an error. Fail loudly instead.
          if (TPGRecord.BYTES_PER_SQUARE_RECORD != 141)
          {
            throw new InvalidOperationException(
              "ConvertToFlatTPG: model expects 141 bytes/square (V3 aux net) but this Ceres build " +
              $"is V2 (BYTES_PER_SQUARE_RECORD = {TPGRecord.BYTES_PER_SQUARE_RECORD}). " +
              "Rebuild Ceres with USE_V3_TPG_RECORD defined (Ceres.Chess.csproj) to serve V3 nets.");
          }
          bytesPerSquareOut = 141;  // V3 141-channel model — pass through
        }
        else
        {
          throw new InvalidOperationException(
            $"ConvertToFlatTPG: unexpected squareValuesByte width per position = {sizePerPos}; " +
            $"expected {64 * 137} (legacy) or {64 * 141} (V3).");
        }
      }
      bool needsSliceForModel = bytesPerSquareOut < TPGRecord.BYTES_PER_SQUARE_RECORD;
      int numConvertedElements = bytesPerSquareOut * 64 * batch.NumPos;

      bool useTemporarySqureValuesByte = squareValuesByte.IsEmpty;
      if (useTemporarySqureValuesByte)
      {
        // Size the temporary for the LARGEST (V3) width even when aux is off, so a
        // single pool covers all three model widths. Default 1024 max batch size.
        squareValuesByteTemporary ??= new byte[TPGRecord.BYTES_PER_SQUARE_RECORD * 64 * 1024];
        if (squareValuesByteTemporary.Length < numConvertedElements)
        {
          squareValuesByteTemporary = new byte[numConvertedElements];
        }
        squareValuesByte = squareValuesByteTemporary;
      }

      if (!needsSliceForModel)
      {
        // Caller's buffer matches the natural compile-time stride
        // (V3/141 with V3=true, or 137 with V3=false). Direct write.
        TPGRecordConverter.ConvertPositionsToRawSquareBytes(batch, includeHistory, batch.Moves, EMIT_PLY_SINCE,
                                                            optionsCeres.QNegativeBlunders, optionsCeres.QPositiveBlunders,
                                                            out _, squareValuesByte, legalMoveIndices);
      }
      else
      {
        // V3 layout (141 bytes/sq) but caller's buffer is sized for the legacy 137-channel
        // model. Run the converter into the full 141-byte-stride temp, then scatter to the
        // narrower stride by copying only the first 137 bytes of each square (dropping aux).
        int SRC_STRIDE = TPGRecord.BYTES_PER_SQUARE_RECORD;  // 141 with V3
        int DST_STRIDE = bytesPerSquareOut;                   // 137 (only narrower option)
        int numFullElements = SRC_STRIDE * 64 * batch.NumPos;
        byte[] fullBuffer = ArrayPool<byte>.Shared.Rent(numFullElements);
        try
        {
          TPGRecordConverter.ConvertPositionsToRawSquareBytes(batch, includeHistory, batch.Moves, EMIT_PLY_SINCE,
                                                              optionsCeres.QNegativeBlunders, optionsCeres.QPositiveBlunders,
                                                              out _, new Memory<byte>(fullBuffer, 0, numFullElements),
                                                              legalMoveIndices);
          Span<byte> outSpan = squareValuesByte.Span;
          for (int p = 0; p < batch.NumPos; p++)
          {
            int srcBase = p * 64 * SRC_STRIDE;
            int dstBase = p * 64 * DST_STRIDE;
            for (int sq = 0; sq < 64; sq++)
            {
              fullBuffer.AsSpan(srcBase + sq * SRC_STRIDE, DST_STRIDE)
                .CopyTo(outSpan.Slice(dstBase + sq * DST_STRIDE, DST_STRIDE));
            }
          }
        }
        finally
        {
          ArrayPool<byte>.Shared.Return(fullBuffer);
        }
      }

      // If we are providing float inputs, then it is necessary to do the
      // (slow) convertion from Half to float (and also divide by 100).
      if (useTemporarySqureValuesByte)
      {
        CopyAndDivide(new Memory<byte>(squareValuesByteTemporary, 0, numConvertedElements),
                      squareValues, TPGSquareRecord.SQUARE_BYTES_DIVISOR);
      }
    }
  }
}
