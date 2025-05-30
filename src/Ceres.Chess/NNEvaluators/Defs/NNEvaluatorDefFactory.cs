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

using Ceres.Chess.NNEvaluators.Specifications;
using Chess.Ceres.NNEvaluators;

#endregion


namespace Ceres.Chess.NNEvaluators.Defs
{
  /// <summary>
  /// Static helper methods to constrct NNEvaluatorDefs.
  /// </summary>
  public static class NNEvaluatorDefFactory
  {
   
    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="netSpecification"></param>
    /// <param name="deviceSpecification"></param>
    /// <param name="evaluatorOptions"></param>
    /// <returns></returns>
    public static NNEvaluatorDef FromSpecification(string netSpecification, 
                                                   string deviceSpecification, 
                                                   NNEvaluatorOptions evaluatorOptions = null)
    {
      ArgumentNullException.ThrowIfNullOrEmpty(netSpecification);
      ArgumentNullException.ThrowIfNullOrEmpty(deviceSpecification);

      NNNetSpecificationString netObj = new NNNetSpecificationString(netSpecification);
      NNDevicesSpecificationString deviceObj = new NNDevicesSpecificationString(deviceSpecification);
      
      NNEvaluatorDef ret = new NNEvaluatorDef(netObj.ComboType, netObj.NetDefs, 
                                              deviceObj.ComboType, deviceObj.Devices, netObj.OptionsString, deviceObj.SharingName);
      ret.OptionsString = netObj.OptionsString;
      ret.Description = netSpecification + " " + deviceSpecification;
      ret.Options = evaluatorOptions;

      return ret;
    }


    public static NNEvaluatorDef SingleNet(string netID, NNEvaluatorType evaluatorType, string sharedName, int[] gpuIDs,
                                           string overrideEngineType)
    {
      return SingleNet(netID, evaluatorType, NNEvaluatorPrecision.FP16, sharedName, overrideEngineType, gpuIDs);
    }


    public static NNEvaluatorDef SingleNet(string netID, NNEvaluatorType evaluatorType,
                                           NNEvaluatorPrecision precision,
                                           string sharedName, string overrideEngineType,
                                           params int[] gpuIDs)
    {
      NNEvaluatorDeviceDef[] devices = new NNEvaluatorDeviceDef[gpuIDs.Length];
      for (int i = 0; i < gpuIDs.Length; i++)
      {
        devices[i] = new NNEvaluatorDeviceDef(NNDeviceType.GPU, gpuIDs[i], overrideEngineType: overrideEngineType);
      }

      NNEvaluatorDeviceComboType type = gpuIDs.Length == 1 ? NNEvaluatorDeviceComboType.Single 
                                                               : NNEvaluatorDeviceComboType.Split;
      return new NNEvaluatorDef(new NNEvaluatorNetDef(netID, evaluatorType, precision), type, sharedName, devices);
    }


    public static NNEvaluatorDef SingleNet(string netID, NNEvaluatorType evaluatorType, string sharedName, 
                                           string overrideEngineType, params (int GPUID, float Fraction)[] gpuIDAndFractions)
    {
      NNEvaluatorNetDef nd = new NNEvaluatorNetDef(netID, evaluatorType);

      (NNEvaluatorDeviceDef device, float fraction)[] deviceWithFractions = new (NNEvaluatorDeviceDef, float)[gpuIDAndFractions.Length];
      for (int i = 0; i < gpuIDAndFractions.Length; i++)
      {
        deviceWithFractions[i] = (new NNEvaluatorDeviceDef(NNDeviceType.GPU, gpuIDAndFractions[i].GPUID), gpuIDAndFractions[i].Fraction);
      }
      return new NNEvaluatorDef(nd, NNEvaluatorDeviceComboType.Split, sharedName, deviceWithFractions);
    }


  }
}
