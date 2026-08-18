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

using System.Diagnostics;
using System.Threading;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// Accumulates the total wall-clock time during which at least one of a set of
  /// evaluators is executing inside the (C++) backend interop boundary ("backend time").
  ///
  /// A single instance is shared by the (up to two) evaluators of an NNEvaluatorSet
  /// so that overlapping backend calls collapse into a single busy interval (i.e.
  /// the value computed is the UNION of the evaluators' busy intervals, not their sum).
  ///
  /// This is computed cheaply via a reference count of how many evaluators are
  /// currently in the backend, without storing per-batch intervals: a busy period
  /// opens when the count rises from 0 and closes when it returns to 0.
  ///
  /// Additionally tracks a breakdown of busy time by concurrency level (exactly one
  /// evaluator in the backend vs two or more) and a histogram of the idle gaps between
  /// busy periods. Together these distinguish the two low-utilization regimes:
  ///   - "clumped" iterators: high dual-overlap fraction AND large/frequent idle gaps
  ///     (both evaluators finish together, then both do CPU work together), vs
  ///   - a well-staggered pipeline: idle gaps near zero (overlap fraction may be
  ///     anything; high overlap alone is fine on a saturated device).
  ///
  /// The complement (total search time minus busy time) is the time during which
  /// NEITHER evaluator is in the backend (i.e. the GPU is idle / pure C# overhead).
  /// </summary>
  public sealed class BackendTimeTracker
  {
    readonly Lock lockObj = new();

    /// <summary>
    /// Number of evaluators currently executing inside the backend.
    /// </summary>
    int busyCount;

    /// <summary>
    /// Timestamp (Stopwatch ticks) of the most recent busyCount transition
    /// (valid only once everUsed).
    /// </summary>
    long lastTransitionTimestamp;

    /// <summary>
    /// Cumulative busy time (Stopwatch ticks) during which exactly one evaluator
    /// was inside the backend.
    /// </summary>
    long soloBusyTicks;

    /// <summary>
    /// Cumulative busy time (Stopwatch ticks) during which two or more evaluators
    /// were inside the backend concurrently.
    /// </summary>
    long dualBusyTicks;

    /// <summary>
    /// Number of completed busy periods (busyCount returned to 0).
    /// </summary>
    long numBusyPeriods;

    /// <summary>
    /// Histogram of idle-gap durations (time between consecutive busy periods).
    /// Bucket upper bounds in milliseconds are given by IDLE_GAP_BUCKET_MS.
    /// </summary>
    public static readonly double[] IDLE_GAP_BUCKET_MS = { 0.1, 0.5, 1, 2, 4, 8, 16, double.PositiveInfinity };
    readonly long[] idleGapCounts = new long[IDLE_GAP_BUCKET_MS.Length];

    /// <summary>
    /// Cumulative idle time (ticks) between busy periods (excludes time before first period).
    /// </summary>
    long idleGapTicks;

    /// <summary>
    /// Whether EnterBackend has been called at least once since the last Reset.
    /// Used to distinguish "supported and measured" from "unsupported backend".
    /// </summary>
    bool everUsed;


    /// <summary>
    /// Resets all accumulated state. Call at the start of each search.
    /// </summary>
    public void Reset()
    {
      lock (lockObj)
      {
        busyCount = 0;
        lastTransitionTimestamp = 0;
        soloBusyTicks = 0;
        dualBusyTicks = 0;
        numBusyPeriods = 0;
        idleGapTicks = 0;
        System.Array.Clear(idleGapCounts);
        everUsed = false;
      }
    }


    /// <summary>
    /// Closes the time segment since the last transition, attributing it to the
    /// bucket for the current busyCount level. Caller must hold lockObj.
    /// </summary>
    void CloseSegment(long now)
    {
      long elapsed = now - lastTransitionTimestamp;
      if (busyCount == 1)
      {
        soloBusyTicks += elapsed;
      }
      else if (busyCount >= 2)
      {
        dualBusyTicks += elapsed;
      }
      lastTransitionTimestamp = now;
    }


    /// <summary>
    /// Records that an evaluator has entered the backend interop boundary.
    /// </summary>
    public void EnterBackend()
    {
      lock (lockObj)
      {
        long now = Stopwatch.GetTimestamp();
        if (busyCount == 0)
        {
          // Record the idle gap just ended (only between busy periods, not before the first).
          if (everUsed)
          {
            long gap = now - lastTransitionTimestamp;
            idleGapTicks += gap;
            double gapMS = gap * 1000.0 / Stopwatch.Frequency;
            for (int b = 0; b < IDLE_GAP_BUCKET_MS.Length; b++)
            {
              if (gapMS <= IDLE_GAP_BUCKET_MS[b])
              {
                idleGapCounts[b]++;
                break;
              }
            }
          }
          lastTransitionTimestamp = now;
        }
        else
        {
          CloseSegment(now);
        }
        busyCount++;
        everUsed = true;
      }
    }


    /// <summary>
    /// Records that an evaluator has exited the backend interop boundary.
    /// </summary>
    public void ExitBackend()
    {
      lock (lockObj)
      {
        long now = Stopwatch.GetTimestamp();
        CloseSegment(now);
        if (--busyCount == 0)
        {
          numBusyPeriods++;
        }
      }
    }


    /// <summary>
    /// Cumulative backend-busy time in seconds (time during which at least one
    /// evaluator was inside the backend).
    ///
    /// Includes the currently-open busy interval (if an evaluator is inside the backend
    /// at the moment of the read). Without this, a snapshot taken while the backend is
    /// continuously occupied (busyCount never returns to 0, as happens under tightly
    /// overlapped evaluators) would report far too little busy time (potentially 0),
    /// since completed intervals are only folded into the accumulators on close.
    /// </summary>
    public double BusySeconds
    {
      get
      {
        lock (lockObj)
        {
          long ticks = soloBusyTicks + dualBusyTicks;
          if (busyCount > 0)
          {
            ticks += Stopwatch.GetTimestamp() - lastTransitionTimestamp;
          }
          return ticks / (double)Stopwatch.Frequency;
        }
      }
    }


    /// <summary>
    /// Busy time in seconds during which exactly one evaluator was inside the backend.
    /// </summary>
    public double SoloBusySeconds
    {
      get
      {
        lock (lockObj)
        {
          return soloBusyTicks / (double)Stopwatch.Frequency;
        }
      }
    }


    /// <summary>
    /// Busy time in seconds during which two or more evaluators were inside the backend.
    /// </summary>
    public double DualBusySeconds
    {
      get
      {
        lock (lockObj)
        {
          return dualBusyTicks / (double)Stopwatch.Frequency;
        }
      }
    }


    /// <summary>
    /// Number of completed busy periods (equivalently, number of idle gaps + 1).
    /// </summary>
    public long NumBusyPeriods
    {
      get
      {
        lock (lockObj)
        {
          return numBusyPeriods;
        }
      }
    }


    /// <summary>
    /// Total idle time (seconds) between busy periods.
    /// </summary>
    public double IdleGapSeconds
    {
      get
      {
        lock (lockObj)
        {
          return idleGapTicks / (double)Stopwatch.Frequency;
        }
      }
    }


    /// <summary>
    /// Returns a copy of the idle-gap histogram counts (buckets per IDLE_GAP_BUCKET_MS).
    /// </summary>
    public long[] IdleGapHistogram
    {
      get
      {
        lock (lockObj)
        {
          return (long[])idleGapCounts.Clone();
        }
      }
    }


    /// <summary>
    /// Human-readable one-line summary of the concurrency breakdown and idle-gap histogram.
    /// </summary>
    public string DetailString
    {
      get
      {
        lock (lockObj)
        {
          double f = Stopwatch.Frequency;
          double solo = soloBusyTicks / f;
          double dual = dualBusyTicks / f;
          double busy = solo + dual;
          double overlapFrac = busy > 0 ? dual / busy : 0;
          var sb = new System.Text.StringBuilder();
          sb.Append($"solo {solo:F2}s  dual {dual:F2}s (overlap {overlapFrac:F2} of busy)  idleGaps n={numBusyPeriods:N0} sum={idleGapTicks / f:F2}s");
          sb.Append("  [ms]");
          for (int b = 0; b < IDLE_GAP_BUCKET_MS.Length; b++)
          {
            string hi = double.IsInfinity(IDLE_GAP_BUCKET_MS[b]) ? "inf" : IDLE_GAP_BUCKET_MS[b].ToString("0.##");
            sb.Append($" <={hi}:{idleGapCounts[b]}");
          }
          return sb.ToString();
        }
      }
    }


    /// <summary>
    /// Whether the tracker has recorded any backend activity since the last Reset
    /// (false indicates the backend does not support this instrumentation).
    /// </summary>
    public bool EverUsed
    {
      get
      {
        lock (lockObj)
        {
          return everUsed;
        }
      }
    }
  }
}
