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
using System.IO;
using System.Collections.Generic;
using System.Linq;

using Ceres.Base.Misc;

using Chess.Ceres.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators.Specifications.Internal;
using Ceres.Chess.UserSettings;

#endregion

namespace Ceres.Chess.NNEvaluators.Specifications
{
  /// <summary>
  /// Manages parsing of a neural net specification string which 
  /// specifies the neural network(s) to be used for evaluating positions.
  /// 
  /// Examples: 
  ///   "LC0:42767"
  ///   "703810#8", 
  ///   "703810#16",
  ///   "LC0:703810=0.5,66193=0.5"
  ///   "LC0:703810,427667"
  ///   "LS15;0.25;0.25;0.25;0.25,66666;0.75;0.75;0.75;0.25" (value, policy, MLH, uncertainty)  
  /// </summary>
  public record NNNetSpecificationString
  {
    /// <summary>
    /// Method used to combine (possibly) multiple nets.
    /// </summary>
    public readonly NNEvaluatorNetComboType ComboType;

    /// <summary>
    /// List of nets to be used, and their fractional weights for the value, policy, MLH and uncertainty heads.
    /// </summary>
    public readonly List<(NNEvaluatorNetDef def, float wtValue, float wtValue2, float wtPolicy, 
                                                 float wtMLH, float wtUncertainty, float wtUncertaintyP)> NetDefs;
   
    /// <summary>
    /// Optional options string (to be passed to the evaluator for interpretation/use).
    /// </summary>
    public readonly string OptionsString;

    /// <summary>
    /// Specification string used as the definition.
    /// </summary>
    public readonly string SpecString;


    /// <summary>
    /// Constructor which parses a net specification string.
    /// </summary>
    /// <param name="netString"></param>
    public NNNetSpecificationString(string netString)
    {
      ArgumentException.ThrowIfNullOrEmpty(netString, nameof(netString)); 

      SpecString = netString;

      // Build network definitions
      (string netOptions, List<(string, NNEvaluatorType, NNEvaluatorPrecision, float, float, float, float, float, float)> parts) netParts = OptionsParserHelpers.ParseNetworkOptions(netString);

      OptionsString = netParts.netOptions;

      NetDefs = new List<(NNEvaluatorNetDef, float, float, float, float, float, float)>();
      foreach (var netSegment in netParts.parts)
      {
        NetDefs.Add((new NNEvaluatorNetDef(netSegment.Item1, netSegment.Item2, netSegment.Item3), netSegment.Item4, netSegment.Item5, netSegment.Item6, 
                                           netSegment.Item7, netSegment.Item8, netSegment.Item9));
      }

      if (NetDefs.Count == 1)
      {
        NetDefs[0].def.ShortID = netString;
      }

      ComboType = NetDefs.Count == 1 ? NNEvaluatorNetComboType.Single : NNEvaluatorNetComboType.WtdAverage;
    }


    /// <summary>
    /// Returns if specified network string is valid.
    /// </summary>
    /// <param name="netString"></param>
    /// <returns></returns>
    public static bool IsValid(string netString)
    {
      try
      {
        new NNNetSpecificationString(netString);
        return true;
      }
      catch
      {
        return false;
      }
    }


    /// <summary>
    /// Returns a readable string representation of the network specification.
    /// </summary>
    /// <param name="net"></param>
    /// <returns></returns>
    static string NetInfoStr((NNEvaluatorNetDef def, float wtValue, float wtValue2, float wtPolicy, float wtMLH, float wtUncertainty, float wtUncertaintyP) net)
    {
      if (net.wtValue == 1)
      {
        return $"{net.def} ";
      }
      else
      {
        return $"({net.def} {net.wtValue} {net.wtValue2} {net.wtPolicy} {net.wtMLH} {net.wtUncertainty} {net.wtUncertaintyP}) ";
      }
    }


    /// <summary>
    /// Returns a readable string representation of the network specification (for the case of multiple networks or non LC0 networks).
    /// </summary>
    /// <param name="comboType"></param>
    /// <param name="nets"></param>
    /// <returns></returns>
    static string ToSpecificationStringComplex(NNEvaluatorNetComboType comboType, IEnumerable<(NNEvaluatorNetDef, float, float, float, float, float, float)> nets)
    {
      string ret = $"Nets: {comboType} ";
      foreach ((NNEvaluatorNetDef, float, float, float, float, float, float) net in nets)
      {
        ret += NetInfoStr(net);
      }
      return ret;
    }


    /// <summary>
    /// Returns a readable string representation of the network specification (for the general case).
    /// </summary>
    /// <param name="comboType"></param>
    /// <param name="nets"></param>
    /// <returns></returns>
    public static string ToSpecificationString(NNEvaluatorNetComboType comboType, IEnumerable<(NNEvaluatorNetDef, float, float, float, float, float, float)> nets)
    {
      // TODO: Currently support conversion back to original specification string only for simple cases
      if (comboType == NNEvaluatorNetComboType.Single && nets.Count() == 1)
      {
        NNEvaluatorType typeOfFirst = nets.First().Item1.Type;
        if (typeOfFirst == NNEvaluatorType.Ceres)
        {
          return $"Network=Ceres:{nets.First().Item1.NetworkID}";
        }
        if (typeOfFirst == NNEvaluatorType.LC0)
        {
          return $"Network=LC0:{nets.First().Item1.NetworkID}";
        }
      }

      return ToSpecificationStringComplex(comboType, nets);
    }

    #region Downloading

    /// <summary>
    /// Attempts to auto-download any constituent networks 
    /// are Ceres nets that are not already found locally.
    /// </summary>
    /// <param name="uciMessagesOnly"></param>
    public bool AutodownloadCeresNetIfNeeded(bool uciMessagesOnly)
    {
      bool anyFailed = false;

      foreach ((NNEvaluatorNetDef def, float wtValue, float wtValue2, float wtPolicy, float wtMLH, float wtUncertainty, float wtUncertaintyP) def in NetDefs)
      {
        if (def.def.Type == NNEvaluatorType.Ceres && !File.Exists(def.def.NetworkID))
        {
          string ceresNetDirectory = CeresUserSettingsManager.Settings.DirCeresNetworks ?? ".";
          string ceresNetID = def.def.NetworkID;

          CeresNetDownloader downloader = new();
          (bool alreadyDownloaded, string downloadedNetworkPath) = downloader.DownloadCeresNetIfNeeded(ceresNetID, ceresNetDirectory, true);
          if (!alreadyDownloaded && downloadedNetworkPath != null)
          {
            Console.WriteLine($"uci info auto download complete");
          }

          if (downloadedNetworkPath == null)
          {
            anyFailed = true;
          } 
        }
      }
      return !anyFailed;
    }

    #endregion
  }
}
