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
using System.Collections.Concurrent;
using System.Diagnostics;

#endregion

namespace Ceres.Chess.External.CEngine
{
  /// <summary>
  /// Tracks child engine processes (e.g. Ceres or Lc0 launched via UCI) and ensures
  /// they are terminated when this (parent) process exits, so that no orphaned engine
  /// processes are left running.
  ///
  /// This is "Layer 1" cleanup: it relies on managed shutdown hooks and therefore covers
  /// the common termination paths -- normal exit, Environment.Exit, an unhandled exception,
  /// Ctrl-C/Ctrl-Break (POSIX SIGINT/SIGQUIT map to these on .NET), and SIGTERM (which runs
  /// graceful shutdown, raising ProcessExit). Cleanup on Ctrl-C runs only after any other
  /// registered CancelKeyPress handler, so a component performing an orchestrated graceful
  /// shutdown (which may block in its handler until drained) is not undercut by an early kill.
  ///
  /// It does NOT cover an unstoppable kill of the parent (SIGKILL / "kill -9", or some
  /// debugger/IDE hard-stops), because in those cases no managed code runs at all. Guarding
  /// against that requires an OS-level mechanism (Windows Job Object / Linux PR_SET_PDEATHSIG)
  /// which is intentionally not implemented here.
  ///
  /// Notes on the APIs used: although additional application domains cannot be created on
  /// modern .NET, AppDomain.CurrentDomain and its ProcessExit / UnhandledException events
  /// remain fully supported on .NET (including .NET 10) and are the documented hooks for
  /// process-lifetime notifications.
  /// </summary>
  public static class ChildProcessGuard
  {
    /// <summary>
    /// Live child processes, keyed by process id. Entries are removed automatically when a
    /// process exits on its own.
    /// </summary>
    static readonly ConcurrentDictionary<int, Process> tracked = new();

    /// <summary>
    /// The Ctrl-C/Ctrl-Break cleanup handler (kept in a field so Track can re-register it,
    /// see MoveCancelHandlerToEndOfChain).
    /// </summary>
    static readonly ConsoleCancelEventHandler cancelKeyHandler = (_, _) => KillAll();

    static ChildProcessGuard()
    {
      // Graceful CLR shutdown (normal return / Environment.Exit), and -- on .NET -- SIGTERM
      // and un-canceled Ctrl-C. This is the only hook needed for actual process termination.
      AppDomain.CurrentDomain.ProcessExit += (_, _) => KillAll();

      // Unhandled exception bringing the process down. On modern .NET this cannot prevent
      // termination; we just use it to clean up the children first.
      AppDomain.CurrentDomain.UnhandledException += (_, _) => KillAll();

      // Ctrl-C / Ctrl-Break in a console (cross-platform; on Unix this also covers SIGINT and
      // SIGQUIT, which .NET maps to CancelKeyPress). We do not set e.Cancel, so if no other
      // handler elects to keep the process alive it still terminates after the children are
      // killed. Components orchestrating a graceful shutdown (e.g. TournamentManager) register
      // handlers which block until their shutdown completes; because this handler is kept LAST
      // in the invocation chain (see MoveCancelHandlerToEndOfChain) it runs only after such an
      // orchestrated drain has finished, rather than killing the engines out from under it.
      //
      // Deliberately NOT registered: PosixSignalRegistration handlers for SIGINT/SIGQUIT.
      // Those fire independently of (and before) the CancelKeyPress chain, which killed the
      // engines immediately upon Ctrl-C even while a graceful tournament shutdown was pending.
      Console.CancelKeyPress += cancelKeyHandler;
    }

    /// <summary>
    /// Re-registers the Ctrl-C cleanup handler so it sits after any handler subscribed since the
    /// last Track call (multicast handlers run in subscription order). This keeps the kill-on-
    /// Ctrl-C as the last resort, after any graceful-shutdown handler has been given the chance
    /// to run (or block) first.
    /// </summary>
    static void MoveCancelHandlerToEndOfChain()
    {
      Console.CancelKeyPress -= cancelKeyHandler;
      Console.CancelKeyPress += cancelKeyHandler;
    }

    /// <summary>
    /// Registers a freshly started child process to be killed when this process exits.
    /// Safe to call for every launched engine process.
    /// </summary>
    /// <param name="process">a started Process</param>
    public static void Track(Process process)
    {
      if (process == null)
      {
        return;
      }

      tracked[process.Id] = process;

      MoveCancelHandlerToEndOfChain();

      // Auto-remove from the registry if the engine exits on its own.
      try
      {
        process.EnableRaisingEvents = true;
        process.Exited += (_, _) => tracked.TryRemove(process.Id, out _);
      }
      catch
      {
        // Process may have already exited; ignore.
      }
    }

    /// <summary>
    /// Terminates all tracked child processes (and their descendants). Idempotent and never throws.
    /// </summary>
    static void KillAll()
    {
      foreach (Process process in tracked.Values)
      {
        try
        {
          if (!process.HasExited)
          {
            process.Kill(entireProcessTree: true);
          }
        }
        catch
        {
          // Best effort: process may have already exited or be inaccessible.
        }
      }
    }
  }
}
