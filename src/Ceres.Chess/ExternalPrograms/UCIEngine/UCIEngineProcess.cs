#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region  Using directives

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

#endregion

namespace Ceres.Chess.External.CEngine
{
  public delegate void ReadEvent(int id, string text);

  /// <summary>
  /// Low-level interface to an external engine running UCI protocol.
  /// </summary>
  public class UCIEngineProcess
  {
    public static bool VERBOSE = false;

    static UCIEngineProcess()
    {
      if (VERBOSE)
      {
        Console.WriteLine("UCIEngineProcess.VERBOSE is true");
      }
    }

    public event ReadEvent ReadEvent;

    public Process EngineProcess = null;
    public bool EngineLoaded { get; private set; }

    public bool HasExited => EngineProcess.HasExited;

    public int EngineID => EngineProcess.Id;

    public string EngineName { set; private get; }

    public StreamReader EngineOutput => EngineProcess.StandardOutput;

    public System.IO.StreamWriter EngineInput => EngineProcess.StandardInput;

    public readonly string EXEPath;
    public readonly string Args;
    public readonly string WorkingDir;
    public readonly Dictionary<string, string> EnvironmentVariables = null;

    string lastCommandSent;


    public UCIEngineProcess(string engineName, string exePath, string args = null, string workingDir = null, Dictionary<string, string> environmentVariables = null)
    {
      EngineName = engineName;
      EXEPath = exePath;
      Args = args;
      WorkingDir = workingDir;
      EngineProcess = new Process();
      EnvironmentVariables = environmentVariables;
    }

    public void SendCommand(string command)
    {
      if (!String.IsNullOrEmpty(command))
      {
        lastCommandSent = command;
        if (VERBOSE)
        {
          Console.WriteLine(EngineID + " SEND: " + command);
        }
        EngineInput.Write(command);
      }
    }

    public void SendCommandLine(string command) => SendCommand(command + "\r\n");

    bool readyOKSeen = false;
    bool haveWarnedWait = false;

    public void WaitForReadyOK(string descString)
    {
      int waitCount = 0;
      while (!readyOKSeen)
      {
        if (EngineProcess.HasExited)
        {
          Console.WriteLine("\r\n");
          Console.WriteLine("EXE         : " + EXEPath);
          Console.WriteLine("Args        : " + Args);
          Console.WriteLine("Working Dir : " + EngineProcess.StartInfo.WorkingDirectory);
          throw new Exception($"Error: the engine process has exited ({descString}) last command was {lastCommandSent}");
        }
        //        else if (lastError != null)
        //          throw new Exception($"UCI error {lastError} ({descString})");

        System.Threading.Thread.Sleep(1);
        if (!haveWarnedWait && waitCount == 30_000)
        {
          Console.WriteLine($"--------------> Warn: waiting >{waitCount}ms for uciok on {descString}");
          haveWarnedWait = true;
        }
        waitCount++;
      }
    }

    public void SendIsReadyAndWaitForOK()
    {
      readyOKSeen = false;
      SendCommandLine("isready");
      WaitForReadyOK("isready");
    }

    public void Shutdown()
    {
      if (EngineLoaded)
      {
        EngineProcess.WaitForExit();
        EngineProcess.Close();
      }
    }

    void ErrorReceviedEvent(object sender, DataReceivedEventArgs e)
    {
      Console.WriteLine($"UCIEngineProcessError: {e.Data}");
    }

    void ReceviedEvent(object sender, DataReceivedEventArgs e)
    {
      if (!String.IsNullOrEmpty(e.Data))
      {
        if (VERBOSE)
        {
          Console.WriteLine(EngineID + " RECEIVE: " + e.Data);
        }

        if (e.Data != null && e.Data.Contains("readyok"))
        {
          readyOKSeen = true;
        }

        if (ReadEvent != null)
        {
          ReadEvent(((Process)sender).Id, e.Data);
        }
      }
    }

    public void StartEngine(bool checkExecutableExists = true)
    {
      if (EngineName == null)
      {
        throw new Exception("EngineName not set");
      }

      if (checkExecutableExists && !File.Exists(EXEPath))
      {
        throw new Exception($"Engine executable {EXEPath} not found");
      }

      EngineProcess.StartInfo.UseShellExecute = false;
      EngineProcess.StartInfo.RedirectStandardInput = true;
      EngineProcess.StartInfo.RedirectStandardOutput = true;
      EngineProcess.StartInfo.RedirectStandardError = true;

      // Isolate the engine from interactive Ctrl-C. Otherwise the engine dies immediately upon
      // Ctrl-C, breaking pipes mid-game while the tournament shutdown handler is still trying to
      // finish gracefully. Orderly termination is via "quit" (TerminateEngine), and
      // ChildProcessGuard still force-kills (SIGKILL) the engine if this process exits,
      // so it cannot be orphaned.
      if (OperatingSystem.IsWindows())
      {
        EngineProcess.StartInfo.FileName = EXEPath;
        EngineProcess.StartInfo.Arguments = Args;

        // A process started in a new process group has Ctrl-C delivery disabled by default,
        // so console Ctrl-C no longer reaches the engine. (This property is Windows-only;
        // setting it on Unix throws PlatformNotSupportedException.)
        EngineProcess.StartInfo.CreateNewProcessGroup = true;
      }
      else
      {
        // On Unix the terminal delivers SIGINT/SIGQUIT to the entire foreground process group,
        // which the engine would inherit membership of. Launch through a shell which marks those
        // signals ignored and then execs the engine: exec preserves the PID (so EngineProcess
        // refers to the engine itself, and kill/HasExited work unchanged) and the ignored
        // disposition survives exec, making the engine immune to terminal Ctrl-C.
        // Deliberately NOT setsid: staying in the same session leaves Linux autogroup
        // (per-session) CPU scheduling identical to launching the engine directly.
        EngineProcess.StartInfo.FileName = "/bin/bash";
        EngineProcess.StartInfo.ArgumentList.Add("-c");
        EngineProcess.StartInfo.ArgumentList.Add($"trap '' INT QUIT; exec \"{EXEPath}\" {Args}");
      }

      // Possibly set provided environment variables
      if (EnvironmentVariables != null)
      {
        foreach (KeyValuePair<string, string> environmentVariable in EnvironmentVariables)
        {
          EngineProcess.StartInfo.EnvironmentVariables.Add(environmentVariable.Key, environmentVariable.Value);
        }
      }

      if (WorkingDir != null)
      {
        EngineProcess.StartInfo.WorkingDirectory = WorkingDir;
      }
      else
      {
        EngineProcess.StartInfo.WorkingDirectory = new FileInfo(EXEPath).DirectoryName;
      }

      if (!VERBOSE)
      {
        EngineProcess.StartInfo.CreateNoWindow = true;
      }

      EngineProcess.OutputDataReceived += new DataReceivedEventHandler(ReceviedEvent);
      EngineProcess.ErrorDataReceived += new DataReceivedEventHandler(ErrorReceviedEvent);

      if (EngineProcess.Start())
      {
        EngineInput.AutoFlush = true;
        EngineLoaded = true;

        // Pin the child engine to the same logical processors as this (parent) process,
        // so the two run with the same NUMA / processor-group locality. 
        if (OperatingSystem.IsWindows())
        {
          try
          {
            EngineProcess.ProcessorAffinity = Process.GetCurrentProcess().ProcessorAffinity;
          }
          catch (Exception exc)
          {
            Console.WriteLine($"Note: could not set child engine processor affinity to match parent: {exc.Message}");
          }
        }

        // Ensure this child engine is terminated if our process exits, so it is not orphaned.
        ChildProcessGuard.Track(EngineProcess);
      }
      else
      {
        throw new Exception($"Engine process start failed for {EngineName}");
      }

    }

    public void TerminateEngine() => SendCommandLine("quit");


    public void ReadAsync()
    {
      EngineProcess.BeginOutputReadLine();

      // Also pump stderr so the child's startup/error output (otherwise silently discarded) is
      // surfaced via ErrorReceviedEvent. This makes failures such as a process that exits before
      // "readyok" diagnosable instead of opaque.
      EngineProcess.BeginErrorReadLine();
    }
  }
}