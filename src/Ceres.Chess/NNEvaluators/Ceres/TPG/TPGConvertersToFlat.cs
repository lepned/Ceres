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
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Ceres.Chess.LC0.Batches;

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

      int numConvertedElements = TPGRecord.BYTES_PER_SQUARE_RECORD * 64 * batch.NumPos;
      bool useTemporarySqureValuesByte = squareValuesByte.IsEmpty;
      if (useTemporarySqureValuesByte)
      {
        squareValuesByteTemporary ??= new byte[137 * 64 * 1024];   // intial guess for max batch size 1024
        if (squareValuesByteTemporary.Length < numConvertedElements)
        {
          squareValuesByteTemporary = new byte[numConvertedElements];
        }
        squareValuesByte = squareValuesByteTemporary;
      }

      NNEvaluatorOptionsCeres optionsCeres = options as NNEvaluatorOptionsCeres;
      // TODO: consider pushing the CopyAndDivide below into this next method
      TPGRecordConverter.ConvertPositionsToRawSquareBytes(batch, includeHistory, batch.Moves, EMIT_PLY_SINCE,
                                                          optionsCeres.QNegativeBlunders, optionsCeres.QPositiveBlunders,
                                                          out _, squareValuesByte, legalMoveIndices);

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
