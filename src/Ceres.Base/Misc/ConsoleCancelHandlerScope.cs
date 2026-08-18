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

#endregion

namespace Ceres.Base.Misc
{
  /// <summary>
  /// Subscribes a handler to Console.CancelKeyPress upon construction
  /// and unsubscribes it upon Dispose.
  ///
  /// Intended for use in a using declaration spanning the operation wanting the handler
  /// (e.g. one tournament or suite run). Because Console.CancelKeyPress is a static event,
  /// a subscription that is never removed roots the handler's closure (and everything
  /// reachable from it, such as an entire completed tournament) for the process lifetime.
  /// </summary>
  public sealed class ConsoleCancelHandlerScope : IDisposable
  {
    ConsoleCancelEventHandler handler;

    public ConsoleCancelHandlerScope(ConsoleCancelEventHandler handler)
    {
      ArgumentNullException.ThrowIfNull(handler);

      this.handler = handler;
      Console.CancelKeyPress += handler;
    }

    public void Dispose()
    {
      if (handler != null)
      {
        Console.CancelKeyPress -= handler;
        handler = null;
      }
    }
  }
}
