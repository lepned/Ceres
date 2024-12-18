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
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using Ceres.Base.DataTypes;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.Evaluators
{
  /// <summary>
  /// Structure that transiently holds the output of a LeafEvaluator
  /// (until the result can be transferred into an MCTS node).
  /// </summary>
  [Serializable]
  [StructLayout(LayoutKind.Sequential, Pack = 2)]
  public struct LeafEvaluationResult
  {
    /// <summary>
    /// Policy win probability percentage.
    /// </summary>
    public FP16 WinP;

    /// <summary>
    /// Policy loss probability percentage.
    /// </summary>
    public FP16 LossP;

    /// <summary>
    /// Optional secondary value head output (W-L).
    /// </summary>
    public FP16 VSecondary;

    /// <summary>
    /// Moves left value (if any).
    /// </summary>
    public FP16 M;

    /// <summary>
    /// Uncertainty of V (scaled).
    /// </summary>
    public byte UncertaintyV;

    /// <summary>
    /// Uncertainty of P (scaled).
    /// </summary>
    public byte UncertaintyP;

    /// <summary>
    /// Transiently holds policy array within which the policy resides
    /// (but will be released after the policy is applied by being copied into a search node)
    /// </summary>
    private Memory<CompressedPolicyVector> policyArray;

    private Memory<CompressedActionVector> actionArray;

    /// <summary>
    /// Index in the policyArray of this policy value.
    /// </summary>
    private short policyArrayIndex;

    /// <summary>
    /// Terminal status of node.
    /// </summary>
    public GameResult TerminalStatus { get; set; }

    /// <summary>
    /// Policy draw probability percentage.
    /// </summary>
    public float DrawP
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        float winP = WinP.ToFloat;
        float lossP = LossP.ToFloat;
        float v = winP - lossP;
        return ParamsSelect.VIsForcedResult(v) ? 0
                                               : 1.0f - (winP + lossP);
      }
    }

    /// <summary>
    /// Value estimate (logistic).
    /// </summary>
    public float V => WinP.ToFloat - LossP.ToFloat;

    /// <summary>
    /// If the underlying policy has been released.
    /// </summary>
    public bool PolicyIsReleased => policyArrayIndex == -1;

    /// <summary>
    /// If the structure has been initialized.
    /// </summary>
    public bool IsNull => TerminalStatus == GameResult.NotInitialized;


    /// <summary>
    /// Constructor from specified values (with no policy).
    /// </summary>
    /// <param name="terminalStatus"></param>
    /// <param name="winP"></param>
    /// <param name="lossP"></param>
    /// <param name="m"></param>
    public LeafEvaluationResult(GameResult terminalStatus, FP16 winP, FP16 lossP, FP16 m, 
                                FP16 vSecondary, byte uncertaintyV, byte uncertaintyP)
    {
      Debug.Assert(terminalStatus != GameResult.NotInitialized);

      TerminalStatus = terminalStatus;
      WinP = winP;
      LossP = lossP;
      VSecondary = vSecondary;
      M = m;
      UncertaintyV = uncertaintyV;
      UncertaintyP = uncertaintyP;
      policyArrayIndex = -1;
      policyArray = null;
      actionArray = null;
    }

    /// <summary>
    /// Re-initializes from specified values (with no policy).
    /// </summary>
    /// <param name="terminalStatus"></param>
    /// <param name="winP"></param>
    /// <param name="lossP"></param>
    /// <param name="m"></param>
    public void Initialize(GameResult terminalStatus, FP16 winP, FP16 lossP, FP16 m, 
                           FP16 vSecondary, byte uncertaintyV, byte uncertaintyP)
    {
      Debug.Assert(terminalStatus != GameResult.NotInitialized);

      TerminalStatus = terminalStatus;
      WinP = winP;
      LossP = lossP;
      VSecondary = vSecondary;
      M = m;
      UncertaintyV = uncertaintyV;
      UncertaintyP = uncertaintyP;
      policyArrayIndex = -1;
      policyArray = null;
      actionArray = null;
    }


    /// <summary>
    /// Constructor from specified values (including policy reference).
    /// </summary>
    /// <param name="terminalStatus"></param>
    /// <param name="winP"></param>
    /// <param name="lossP"></param>
    /// <param name="m"></param>
    /// <param name="policyArray"></param>
    /// <param name="policyArrayIndex"></param>
    public LeafEvaluationResult(GameResult terminalStatus, FP16 winP, FP16 lossP, FP16 m, 
                                FP16 vSecondary, byte uncertaintyV, byte uncertaintyP,
                                Memory<CompressedPolicyVector> policyArray, 
                                Memory<CompressedActionVector> actionArray, short policyArrayIndex)
    {
      Debug.Assert(terminalStatus != GameResult.NotInitialized);

      TerminalStatus = terminalStatus;
      WinP = winP;
      LossP = lossP;
      VSecondary = vSecondary;
      M = m;
      UncertaintyP = uncertaintyP;
      UncertaintyV = uncertaintyV;

      this.policyArrayIndex = policyArrayIndex;
      this.policyArray = policyArray;
      this.actionArray = actionArray;
    }


    /// <summary>
    /// Reference to underlying policy.
    /// </summary>
    public ref readonly CompressedPolicyVector PolicyRef
    {
      get
      {
        if (policyArrayIndex == -1)
        {
          throw new Exception("Internal error: access to release policy object");
        }
        return ref policyArray.Span[policyArrayIndex];
      }
    }


    /// <summary>
    /// Reference to underlying actions.
    /// </summary>
    public ref readonly CompressedActionVector ActionsRef
    {
      get
      {
        if (policyArrayIndex == -1)
        {
          throw new Exception("Internal error: access to release action object");
        }
        return ref actionArray.Span[policyArrayIndex];
      }
    }


    /// <summary>
    /// Memory reference to underlying actions.
    /// </summary>
    public (Memory<CompressedActionVector> actions, int index) ActionInArray
    {
      get => (actionArray, policyArrayIndex);

      set
      {
        Debug.Assert((policyArrayIndex == -1 || policyArrayIndex == value.index) && value.index != -1); // possibly already set by policy
        this.actionArray = value.actions;
        this.policyArrayIndex = (short)value.index;
      }
    } 

    /// <summary>
    /// Memory reference to underlying policy.
    /// </summary>
    public (Memory<CompressedPolicyVector> policies, int index) PolicyInArray
    {
      get => (policyArray, policyArrayIndex);

      set
      {
        Debug.Assert((policyArrayIndex == -1 || policyArrayIndex == value.index) && value.index != -1); // possibly already set by action
        this.policyArray = value.policies;
        this.policyArrayIndex = (short)value.index;
      }
    }


    public CompressedActionVector ActionSingle
    {
      set
      {
        Debug.Assert(policyArrayIndex == -1);
        actionArray = new CompressedActionVector[1] { value };
        policyArrayIndex = 0;
      }
    } 

    /// <summary>
    /// Policy as a CompressedPolicyVector.
    /// </summary>
    public CompressedPolicyVector PolicySingle
    {
      set
      {
        Debug.Assert(policyArrayIndex == -1);
        policyArray = new CompressedPolicyVector[1] { value };
        policyArrayIndex = 0;
      }
    }

    /// <summary>
    /// Releases underlying policy value.
    /// </summary>
    public void ReleasePolicyActionsValues()
    {
      policyArray = null;
      actionArray = null;
      policyArrayIndex = -1;
    }


    /// <summary>
    /// Returns string representation.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      string policyStr = PolicyIsReleased ? "(null)" : PolicyRef.ToString();
      return $"<LeafEvaluationResult {(IsNull ? "(null)>" : $"{TerminalStatus} V={WinP - LossP,6:F3} Policy={policyStr}>")}";
    }

  }

}
