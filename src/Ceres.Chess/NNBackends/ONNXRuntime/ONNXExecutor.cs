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
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Linq;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using Onnx;

using Ceres.Base.Misc;
using Ceres.Base.CUDA;
using Ceres.Base.Benchmarking;

#endregion

/// <summary>
/// Manages evaluation of neural networks using ONNX runtime.
/// 
/// Although ONNX the documentation stats that multiple threads can invoke the Run() method
/// on the same inference session object, we have single-instance buffers for inputs and outputs
/// and therefore take locks to enforce single-threaded access.
/// 
/// TODO: some of the clients of this class could possibly pass in a restricted list of outputNames
///       to eliminate overhead of retrieving values for outputs which moray not be needed in some situations.
/// </summary>
namespace Ceres.Chess.NNBackends.ONNXRuntime
{
  public class ONNXExecutor : IDisposable
  {
    /// <summary>
    /// Name of underlying ONNX file;
    /// </summary>
    public readonly string ONNXFileName;

    /// <summary>
    /// Underlying ONNX runtime session
    /// </summary>
    public readonly InferenceSession Session;

    /// <summary>
    /// ID (index) of GPU to use
    /// </summary>
    public readonly int GPUID;

    /// <summary>
    /// Precision width to use (32=FP32, 16=FP16, 8=FP8).
    /// </summary>
    public int PrecisionNumBits;

    /// <summary>
    /// Minimum supported batch size.
    /// </summary>
    public int MinBatchSize;

    /// <summary>
    /// If TensorRT execution provider should be used.
    /// </summary>
    public bool UseTensorRT;

    /// <summary>
    /// If all outputs should be retained.
    /// </summary>
    public bool RetainRawInputs;


    /// <summary>
    /// The names of the sub-networks if the net is a specially prepared 
    /// Ceres multinet network (containing the string "multinet" in the file name).
    /// </summary>
    public readonly string[] MultiNetNames;


    /// The weights to be used for inference of the sub-networks if the net is a specially 
    /// prepared Ceres multinet network (containing the string "multinet" in the file name).
    public readonly float[] MultiNetWeights;

    /// <summary>
    /// Name of the LoRA adapter file (if any).
    /// </summary>
    public readonly string LoRAAdapterFileName;

    /// <summary>
    /// Execution is serialized by this lock object.
    /// Technically, ONNX runtime sessions are thread-safe
    /// so this might not be strictly necessary.
    /// </summary>
    readonly object lockObject = new object();

    int maxBatchSize;

    RunOptions runOptions;
    bool disposed;

    public readonly IReadOnlyDictionary<string, NodeMetadata> InputsMetadata;

    public enum ONNXInputTypeEnum { Float32, Float16, Byte };


    (string name, NodeMetadata metadata, bool isKnownShape, Float16[] value)[] inputBuffers16;
    (string name, NodeMetadata metadata, bool isKnownShape, byte[] value)[] inputBuffersByte;
    (string name, NodeMetadata metadata, bool isKnownShape, Float16[] value)[] outputBuffers16;
    (string name, NodeMetadata metadata, bool isKnownShape, float[] value)[] inputBuffers32;
    (string name, NodeMetadata metadata, bool isKnownShape, float[] value)[] outputBuffers32;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="shortID"></param>
    /// <param name="onnxFileName"></param>
    /// <param name="onnxModelBytes"></param>
    /// <param name="inputNames"></param>
    /// <param name="nonBatchDimensions"></param>
    /// <param name="precisionNumBits"></param>
    /// <param name="gpuID"></param>
    /// <param name="useTensorRT"></param>
    /// <param name="minBatchSize"></param>
    /// <param name="maxBatchSize"></param>
    /// <param name="enableProfiling"></param>
    /// <param name="retainRawOutputs"></param>
    /// <param name="outputNamesToRetrieve"></param>
    /// <param name="loraAdapterFileName"></param>
    /// <exception cref="Exception"></exception>
    public ONNXExecutor(string shortID,
                        string onnxFileName,
                        byte[] onnxModelBytes,
                        string[] inputNames,
                        string nonBatchDimensions,
                        int precisionNumBits,
                        int gpuID,
                        bool useTensorRT,
                        int minBatchSize,
                        int maxBatchSize,
                        bool enableProfiling,
                        bool retainRawOutputs,
                        string[] outputNamesToRetrieve = null,
                        string loraAdapterFileName = null)
    {
      if (precisionNumBits != 32 && precisionNumBits != 16)
      {
        throw new NotImplementedException();
      }

      ONNXFileName = onnxFileName;
      if (onnxFileName == null && onnxModelBytes == null)
      {
        throw new Exception("Must specify either onnxFileName or onnxModelBytes");
      }

      if (!File.Exists(onnxFileName))
      {
        throw new Exception("ONNX file not found: " + onnxFileName);
      }

      this.maxBatchSize = maxBatchSize;
      string directoryName = onnxFileName == null ? Path.GetTempPath() : new FileInfo(onnxFileName).DirectoryName;

      if (onnxModelBytes == null)
      {
        onnxModelBytes = File.ReadAllBytes(onnxFileName);
      }

      // Recognize if this is a Ceres "multinet" net and 
      // extract metadata (names and weights) if so.
      if (onnxFileName == null || onnxFileName.ToUpper().Contains("MULTINET"))
      {
        using (new TimingBlock("ONNX ModelProto parse"))
        {
          ModelProto onnxProto = ModelProto.Parser.ParseFrom(onnxModelBytes);
//          bool usesSquaresBytes = Ceres.Base.Misc.ONNX.ONNXHelpers.GetMetadataValue(onnxProto, "uses_square_bytes_input") != null;
          string multinetNames = Ceres.Base.Misc.ONNX.ONNXHelpers.GetMetadataValue(onnxProto, "Ceres_multinet_names");
          if (multinetNames != null)
          {
            MultiNetNames = multinetNames.Split(',');
          }

          string multinetWeights = Ceres.Base.Misc.ONNX.ONNXHelpers.GetMetadataValue(onnxProto, "Ceres_multinet_weights");
          if (multinetWeights != null)
          {
            MultiNetWeights = multinetWeights.Split(',').Select(float.Parse).ToArray();
          }

          Console.WriteLine();
          Console.Write("LOADING Ceres MultiNet:");
          for (int i = 0; i < MultiNetNames.Length; i++)
          {
            Console.Write(" " + MultiNetNames[i] + "=" + MultiNetWeights[i]);
          }
        }
      }

      runOptions = new RunOptions();

      if (loraAdapterFileName != null)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, "Install LoRA adapter " + loraAdapterFileName);
        OrtLoraAdapter adapterCeres = OrtLoraAdapter.Create(loraAdapterFileName, null);
        runOptions.AddActiveLoraAdapter(adapterCeres);
      }


      GPUID = gpuID;
      PrecisionNumBits = precisionNumBits;

      MinBatchSize = minBatchSize;
      UseTensorRT = useTensorRT;
      RetainRawInputs = retainRawOutputs;
      LoRAAdapterFileName = loraAdapterFileName;

      // On Linux it was found necessary to touch the instance before any of the operations below
      // to prevent error about a session object not being created.
      // https://github.com/microsoft/onnxruntime/issues/11572
      OrtEnv ortInstance = OrtEnv.Instance();
      SessionOptions so = default;

      //        so.AppendExecutionProvider_CoreML();


      if (gpuID < 0) // CPU. TO DO: clean this up
      {
        so = new SessionOptions();
      }
      else if (useTensorRT)
      {
        const bool USE_DML = false; // This likely requires different ONNXRuntime nuget package
        if (USE_DML)
        {
          so = new SessionOptions();
          so.AppendExecutionProvider_DML(gpuID);
          so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        }
        else
        {
          OrtTensorRTProviderOptions trtProviderOptions = new OrtTensorRTProviderOptions();
          // TODO: this code has no effect for unknown reasons.
          Dictionary<string, string> providerOptionsDict = new();
          providerOptionsDict["device_id"] = gpuID.ToString();
          providerOptionsDict["trt_max_workspace_size"] = "4294967296";

          if (inputNames != null)
          {
            string MakeShapeStr(int size) // construct a shape string of the form expected by the TensorRT execution provider.
            {
              bool firstTime = true;
              string ret = "";
              foreach (string inputName in inputNames)
              {
                if (!firstTime)
                {
#if DEBUG
                  // Testing with TensorRT 10.3 suggests caching fails if there is more than one input.
                  ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "WARNING as workaround for caching failure in ONNXRuntime,"
                    + " shapes string will omit " + inputName);
#endif
                  continue;
                }

                ret += (firstTime ? "" : ",") + inputName + $":{size}x{nonBatchDimensions}";
                firstTime = false;
              }
              return ret;
            }

            // If no profile is specified then TensorRT will build a new profile/engine
            // each time it sees a different batch size.
            // Although possibly faster to execute, these rebuilds would be far too slow.
            // Therefore we build a profile specifying the min, opt, and max batch sizes.
            //
            // Note that the actual performance is seemingly sensitive to the min/max batch sizes
            // in unpredictable ways. Seemingly large max batch sizes (e.g. over about 200)
            // often reduce performance.
            const bool USE_OMNI_PROFILE = true;
            if (USE_OMNI_PROFILE)
            {
              providerOptionsDict["trt_profile_min_shapes"] = MakeShapeStr(minBatchSize);// + "," + MakeShapeStr(128);
              providerOptionsDict["trt_profile_opt_shapes"] = MakeShapeStr(128);// + "," + MakeShapeStr(512);
              providerOptionsDict["trt_profile_max_shapes"] = MakeShapeStr(maxBatchSize);// + "," + MakeShapeStr(1024); // N.B. see note above
            }
          }
#if NOT
Comments from onnxruntime source code:
 /*
   * Parse explicit min/max/opt profile shapes from provider options.
   *
   * The format of min/max/opt profile shapes is defined as below:
   * "input1:dim1xdim2...,input2:dim1xdim2...,...,input1:dim3xdim4...,input2:dim3xdim4...,..."
   *
   * (Note: if multiple shapes with same input name are specified, TRT EP will consider them as multiple profiles.
   *  Please refer to ParserProfileShapes() for more details)
   *
   */
#endif

          // Use timing and engine caches, located in a folder specific to this host.
          string trtSubdirectory;
          bool EMBED = Environment.GetEnvironmentVariable("EMBED_TRT") == "1";
          if (EMBED)
          {
            // "In the case of dumping context model and for security purpose,"
            // "the trt_engine_cache_path should be set with a relative path."
            providerOptionsDict["trt_ep_context_file_path"] = "./";
            providerOptionsDict["trt_dump_ep_context_model"] = "1";
            providerOptionsDict["trt_ep_context_embed_mode"] = "1";

            Console.WriteLine();
            ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "NOTE: EMBED_TRT is set to 1. TensorRT engine will be embedded in the ONNX file _ctx.onnx.");
            ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "NOTE: the _ctx.onnx file will only be created only upon normal termination of this process.");
            ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "NOTE: For security reasons, this file is emitted in subdirectory trt_engines_embed of the working directory of this process.");
            Console.WriteLine();
          }

          trtSubdirectory = Path.Combine(directoryName, "trt_engines", Environment.MachineName);
          Directory.CreateDirectory(trtSubdirectory);
          Console.WriteLine("TensorRT engines will be cached in: " + trtSubdirectory);

          providerOptionsDict["trt_engine_cache_path"] = trtSubdirectory;
          providerOptionsDict["trt_timing_cache_path"] = trtSubdirectory;

          providerOptionsDict["trt_engine_cache_enable"] = "1";
          providerOptionsDict["trt_timing_cache_enable"] = "1";
          //providerOptionsDict["trt_force_timing_cache"] = "true";

          providerOptionsDict["trt_engine_cache_prefix"] = FileUtils.FileNameSanitized(shortID);

          //          providerOptionsDict["trt_detailed_build_log"] = "1";

          if (PrecisionNumBits == 16)
          {
            providerOptionsDict["trt_fp16_enable"] = "1";
          }

          // N.B. Using values other than "4" possibly causes engine generation
          //      to randomly fail to respect the request for FP16, resulting in much slower engine.
          // NOTE: if we knew in advance this ONNX file already has embedded TensorRT, then
          //        "It's suggested to set the ORT graph optimization level to 0"
          providerOptionsDict["trt_builder_optimization_level"] = "4";


          // TODO: For graphs: use IO / Binding to bind input tensors in GPU memory.
          // See: https://github.com/microsoft/onnxruntime/issues/20050
          // "During inference, copy input to same address(input shape shall be the same) of the input used in the first inference run."
          //https://github.com/microsoft/onnxruntime/blob/4a196d15940b0f328735c888e2e861d67602ffcf/onnxruntime/python/tools/transformers/io_binding_helper.py#L212-L307
          // providerOptionsDict["trt_cuda_graph_enable"] = "1"; // NOTE: may fail or yield bad output, requires entire graph to map onto ONNX nodes (?)
          // providerOptionsDict["trt_auxiliary_streams"] = "0";

          //providerOptionsDict["trt_context_memory_sharing_enable"]= "1"; returns error, not obviously faster
          providerOptionsDict["trt_layer_norm_fp32_fallback"] = "1"; // possibly necessary otherwise terrible accuracy

          trtProviderOptions.UpdateOptions(providerOptionsDict);
          so = SessionOptions.MakeSessionOptionWithTensorrtProvider(trtProviderOptions);
        }
      }
      else
      {
        // https://tomwildenhain-microsoft.github.io/onnxruntime/docs/execution-providers/CUDA-ExecutionProvider.html
        OrtCUDAProviderOptions cudaProviderOptions = new();

        Dictionary<string, string> providerOptionsDict = new();
        providerOptionsDict["device_id"] = gpuID.ToString();

        cudaProviderOptions.UpdateOptions(providerOptionsDict);
        so = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);
      }

#if NOT // DEBUG
      so.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
#else
      so.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
#endif

      bool VERBOSE = false;
      if (VERBOSE)
      {
        so.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
        so.LogVerbosityLevel = 999;
        so.LogId = "ort.log.txt";
      }


      if (enableProfiling)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, @"****************   NetExecutorONNXRuntime is profiling to c:\temp ....   ****************");
        so.EnableProfiling = true;
        so.ProfileOutputPathPrefix = @"c:\temp";
      }

      // N.B. A random problem with generated engines being slow (ignored FP16 mode)
      //      may possibly be caused by using ORT_ENABLE_ALL, so instead we use just ORT_ENABLE_EXTENDED.
      so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;


      // N.B. Do not use ORT_PARALLEL, this causes failures in ONNXRuntime when using GPUs with index other than 0.
      so.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;

      // Take a global lock during session creation for two reasons:
      //   - we don't want to try to create TensorRT engines for the same network simultaneously
      //     because this would interfere with performance profiling,
      //     and the second time a cached version should just be used rather than recreated
      //   - reduce likelihood of conflict with other initialization (e.g. other threads using ManagedCUDA).
      lock (CUDADevice.InitializingCUDAContextLockObj)
      {
        using (new TimingBlock($"ONNX InferenceSession create on model of size {onnxModelBytes.Length:N0} bytes"))
        {

          Session = new InferenceSession(onnxModelBytes, so);
          InputsMetadata = Session.InputMetadata;

          // Rewrite the inputsMetadata to ignore any input
          // having one of its dimensions 0.
          // These occur (only?) for inputs which will be used with MultiLoRA adapter feature.
          // The values of those nodes are not explicitly provided upon calls to Run,
          // but rather the ONNX runtime system will fill them in using active LoRA adapters
          // specified in the RunOptions (if any).
          Dictionary<string, NodeMetadata> newDict = new();
          foreach (KeyValuePair<string, NodeMetadata> kvp in InputsMetadata)
          {
            if (!kvp.Value.Dimensions.Contains(0))
            {
              newDict.Add(kvp.Key, kvp.Value);
            }
          }
          InputsMetadata = newDict;

          // Create input and output buffers.
          if (PrecisionNumBits == 32)
          {
            inputBuffers32 = ONNXHelpers.CreateBuffers<float>(InputsMetadata, maxBatchSize);
            outputBuffers32 = ONNXHelpers.CreateBuffers<float>(Session.OutputMetadata, maxBatchSize, outputNamesToRetrieve);
          }
          else if (PrecisionNumBits == 16)
          {
            inputBuffersByte = ONNXHelpers.CreateBuffers<byte>(InputsMetadata, maxBatchSize);
            inputBuffers16 = ONNXHelpers.CreateBuffers<Float16>(InputsMetadata, maxBatchSize);
            outputBuffers16 = ONNXHelpers.CreateBuffers<Float16>(Session.OutputMetadata, maxBatchSize, outputNamesToRetrieve);
          }
          else
          {
            throw new Exception("Unsupported precision (" + PrecisionNumBits + ")");
          }

        }
      }
    }


    bool haveWarned = false;

    /// <summary>
    /// Returns the number of inputs as reported by the ONNX model metadata.
    /// </summary>
    public int NumInputs => InputsMetadata.Count;


    /// <summary>
    /// Common validation and input processing logic shared by both Run method overloads.
    /// </summary>
    /// <typeparam name="T">The input data type (byte or Half)</typeparam>
    /// <param name="inputs">Input array containing memory, shape, and metadata</param>
    /// <param name="batchSize">Batch size for processing</param>
    /// <returns>Processed input data with names and element counts</returns>
    private (Memory<T> input, int[] shape, string inputName, int numElements)[] ValidateAndProcessInputs<T>((Memory<T> input, int[] shape)[] inputs, int batchSize)
    {
      if (InputsMetadata.Count != inputs.Length)
      {
        throw new ArgumentException($"Expected {InputsMetadata.Count} inputs, received " + inputs.Length);
      }

      if (inputs[0].shape[0] > maxBatchSize)
      {
        throw new ArgumentException($"Batch size {inputs[0].shape[0]} exceeds maximum of {maxBatchSize}");
      }

      var inputsONNX = new (Memory<T> input, int[] shape, string inputName, int numElements)[InputsMetadata.Count];

      if (InputsMetadata.Count != 1)
      {
        if (!haveWarned)
        {
          // data type check below is only on first element
          Console.WriteLine("WARNING: Currently only single input ONNX files supported definitively.");
          haveWarned = true;
        }
      }

      int inputIndex = 0;
      foreach (KeyValuePair<string, NodeMetadata> iv in InputsMetadata)
      {
        (Memory<T> input, int[] shape) = inputs[inputIndex];
        string inputName = iv.Key;
        if (inputName == null)
        {
          throw new Exception("Unable to retrieve name of input");
        }

        int numElements = ONNXHelpers.ProductDimensions(shape, batchSize);
        Debug.Assert(input.Length == numElements); // caller to have passed the correct size

        inputsONNX[inputIndex] = (input, shape, inputName, numElements);
        inputIndex++;
      }

      return inputsONNX;
    }

    /// <summary>
    /// Evaluates the input.
    /// </summary>
    /// <param name="inputType"></param>
    /// <param name="inputs"></param>
    /// <param name="batchSize"></param>
    /// <returns></returns>
    public List<(string, Memory<Float16>)> Run(ONNXInputTypeEnum inputType, (Memory<byte> input, int[] shape)[] inputs, int batchSize)
    {
      var inputsONNX = ValidateAndProcessInputs(inputs, batchSize);

      // TODO: Actually the precision of the network is defined by the net itself.
      //       So the inputIsFloat above should be used to determine this, and
      //       the caller should not be offered the chance to set the precision here
      //       (unless we decided to support auto-conversion of ONNX files here).
      if (inputType == ONNXInputTypeEnum.Byte)
      {
        return RunInputByteOutputFloat16(inputsONNX, batchSize);
      }
      else
      {
        throw new NotImplementedException("Unexpected ONNXInputTypeEnum" + inputType);
      }
    }

    /// <summary>
    /// Evaluates the input.
    /// </summary>
    /// <param name="inputType"></param>
    /// <param name="inputs"></param>
    /// <param name="batchSize"></param>
    /// <returns></returns>
    public List<(string, Memory<Float16>)> Run(ONNXInputTypeEnum inputType, (Memory<Half> input, int[] shape)[] inputs, int batchSize)
    {
      var inputsONNX = ValidateAndProcessInputs(inputs, batchSize);

      // TODO: Actually the precision of the network is defined by the net itself.
      //       So the inputIsFloat above should be used to determine this, and
      //       the caller should not be offered the chance to set the precision here
      //       (unless we decided to support auto-conversion of ONNX files here).
      if (inputType == ONNXInputTypeEnum.Float16)
      {
        return RunInputHalfOutputFloat16(inputsONNX, batchSize);
      }
      else if (inputType == ONNXInputTypeEnum.Float32)
      {
        return RunOutputFloat(inputsONNX, batchSize);
      }
      else if (inputType == ONNXInputTypeEnum.Byte)
      {
        throw new Exception("Use the overloaded function for Byte instead");
      }
      else
      {
        throw new NotImplementedException("Unknown ONNXInputTypeEnum" + inputType);
      }
    }



    /// <summary>
    /// 
    /// 
    /// TO DO: eventually we could have a separate (more efficient) code path 
    ///        which is FP16 throughout rather than multiple conversions.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="shape"></param>
    /// <param name="inputName"></param>
    /// <param name="numElements"></param>
    /// <returns></returns>
    internal List<(string, Memory<Float16>)> RunInputByteOutputFloat16((Memory<byte> input, int[] shape, string inputName, int numElements)[] inputs, int batchSize)
    {
      if (batchSize < MinBatchSize)
      {
        throw new ArgumentException($"Batch size {batchSize} is less than minimum of {MinBatchSize}");
      }

      List<NamedOnnxValue> inputsONNX = new(inputs.Length);

      for (int i = 0; i < inputs.Length; i++)
      {
        (Memory<byte> input, int[] shape, string inputName, int numElements) = inputs[i];

        // Cast float inputs directly into the target byte ONNX buffer
        input.Span.CopyTo(inputBuffersByte[i].value);
      }

      (string[] names, OrtValue[] values) inputBuffers = ONNXHelpers.CreateOrtValues(batchSize, inputBuffersByte);

      return RunOutputFloat16Inner<byte>(inputs, inputBuffers, batchSize);
    }

    /// <summary>
    /// 
    /// 
    /// TO DO: eventually we could have a separate (more efficient) code path 
    ///        which is FP16 throughout rather than multiple conversions.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="shape"></param>
    /// <param name="inputName"></param>
    /// <param name="numElements"></param>
    /// <returns></returns>
    internal List<(string, Memory<Float16>)> RunInputHalfOutputFloat16((Memory<Half> input, int[] shape, string inputName, int numElements)[] inputs, int batchSize)
    {
      if (batchSize < MinBatchSize)
      {
        throw new ArgumentException($"Batch size {batchSize} is less than minimum of {MinBatchSize}");
      }

      List<NamedOnnxValue> inputsONNX = new(inputs.Length);

      for (int i = 0; i < inputs.Length; i++)
      {
        (Memory<Half> input, int[] shape, string inputName, int numElements) = inputs[i];

        // Cast float inputs directly into the target Float16 ONNX buffer
        Span<Half> inputBufferSpanHalf = MemoryMarshal.Cast<Float16, Half>(inputBuffers16[i].value);
        input.Span.CopyTo(inputBufferSpanHalf);
      }

      (string[] names, OrtValue[] values) inputBuffers = ONNXHelpers.CreateOrtValues(batchSize, inputBuffers16);

      return RunOutputFloat16Inner<Half>(inputs, inputBuffers, batchSize);
    }



    internal List<(string, Memory<Float16>)> RunOutputFloat16Inner<T>(
      (Memory<T> input, int[] shape, string inputName, int numElements)[] inputs,
      (string[] names, OrtValue[] values) inputBuffers, int batchSize)
    {
      List<(string, Memory<Float16>)> resultArrays = new(outputBuffers16.Length);

      lock (lockObject)
      {
        (string[] names, OrtValue[] values) outputBuffers = ONNXHelpers.CreateOrtValues(batchSize, outputBuffers16);

        bool unknownShapeExists = outputBuffers16.Any(b => !b.isKnownShape);

        if (RetainRawInputs || unknownShapeExists)
        {
          List<string> allOutputNames = outputBuffers16.Select(p => p.name).ToList();
          IDisposableReadOnlyCollection<OrtValue> rr = Session.Run(runOptions, inputBuffers.names, inputBuffers.values, allOutputNames);

          // Dispose of input buffers used.
          foreach (OrtValue v in inputBuffers.values)
          {
            v.Dispose();
          }
          foreach (OrtValue v in outputBuffers.values)
          {
            v.Dispose();
          }

          // Extract results from output buffers.
          for (int i = 0; i < outputBuffers.names.Length; i++)
          {

            if (rr[i].GetTensorTypeAndShape().ElementDataType == TensorElementType.Float16)
            {
              ReadOnlySpan<Float16> data1 = rr[i].GetTensorDataAsSpan<Float16>();
              resultArrays.Add((allOutputNames[i], data1.ToArray()));
            }
            else
            {
              // for now, do not process if data type other than Float16
              // TODO: improve this
              resultArrays.Add((allOutputNames[i], null));
            }

            rr[i].Dispose();
          }
        }
        else
        {
          // Note that IOBinding is not used. As noted in the ONNX documentation,
          // there is not necessarily any benefit of using IOBinding over this simpler
          // method of passing the OrtValue inputs and outputs directly to the Run method.
          Session.Run(runOptions,
                      inputBuffers.names, inputBuffers.values,
                      outputBuffers.names, outputBuffers.values);

          foreach ((string name, NodeMetadata metadata, bool isKnownShape, Float16[] value) resultItem in outputBuffers16)
          {
            // Create a Memory over the ONNX buffer sized to the actual number of elements for this batch.
            Memory<Float16> memory = new Memory<Float16>(resultItem.value)[..ONNXHelpers.ProductDimensions(resultItem.metadata.Dimensions, batchSize)];
            resultArrays.Add((resultItem.name, memory));
          }
        }
      }

      return resultArrays;
    }


    /// <summary>
    /// Runs the network with float inputs (instead of Half).
    /// 
    /// Note that this accepts Half inputs and returns Half inputs, 
    /// but merely upcasts them to floats for ONNX runtime execution and then downcasts results.
    /// 
    /// This is inefficient and does not fully exploit the higher precision of float over Half
    /// (but is intended mostly for debugging purposes).
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="batchSize"></param>
    /// <returns></returns>
    private List<(string, Memory<Float16>)> RunOutputFloat((Memory<Half> input, int[] shape, string inputName, int numElements)[] inputs, int batchSize)
    {
      if (batchSize < MinBatchSize)
      {
        throw new ArgumentException($"Batch size {batchSize} is less than minimum of {MinBatchSize}");
      }

      // Convert inputs to float buffers using pre-allocated inputBuffers32.
      for (int i = 0; i < inputs.Length; i++)
      {
        (Memory<Half> input, int[] shape, string inputName, int numElements) = inputs[i];
        Span<float> inputBufferSpanFloat = inputBuffers32[i].value.AsSpan(0, numElements);
        TensorPrimitives.ConvertToSingle(input.Span, inputBufferSpanFloat);
      }

      List<(string, Memory<Float16>)> resultArrays = new(outputBuffers32.Length);

      lock (lockObject)
      {
        (string[] names, OrtValue[] values) inputBuffers = ONNXHelpers.CreateOrtValues(batchSize, inputBuffers32);
        (string[] names, OrtValue[] values) outputBuffers = ONNXHelpers.CreateOrtValues(batchSize, outputBuffers32);

        bool unknownShapeExists = outputBuffers32.Any(b => !b.isKnownShape);

        if (RetainRawInputs || unknownShapeExists)
        {
          List<string> allOutputNames = outputBuffers32.Select(p => p.name).ToList();
          IDisposableReadOnlyCollection<OrtValue> rr = Session.Run(runOptions, inputBuffers.names, inputBuffers.values, allOutputNames);

          // Dispose of input buffers used.
          foreach (OrtValue v in inputBuffers.values)
          {
            v.Dispose();
          }
          foreach (OrtValue v in outputBuffers.values)
          {
            v.Dispose();
          }

          // Extract results from output buffers.
          for (int i = 0; i < outputBuffers.names.Length; i++)
          {
            if (rr[i].GetTensorTypeAndShape().ElementDataType == TensorElementType.Float)
            {
              ReadOnlySpan<float> data = rr[i].GetTensorDataAsSpan<float>();
              Float16[] halfValues = new Float16[data.Length];
              for (int j = 0; j < data.Length; j++)
              {
                halfValues[j] = (Float16)data[j];
              }
              resultArrays.Add((allOutputNames[i], halfValues));
            }
            else
            {
              // for now, do not process if data type other than float
              resultArrays.Add((allOutputNames[i], null));
            }
            rr[i].Dispose();
          }
        }
        else
        {
          // Note that IOBinding is not used. As noted in the ONNX documentation,
          // there is not necessarily any benefit of using IOBinding over this simpler
          // method of passing the OrtValue inputs and outputs directly to the Run method.
          Session.Run(runOptions,
                      inputBuffers.names, inputBuffers.values,
                      outputBuffers.names, outputBuffers.values);

          foreach ((string name, NodeMetadata metadata, bool isKnownShape, float[] value) resultItem in outputBuffers32)
          {
            int count = ONNXHelpers.ProductDimensions(resultItem.metadata.Dimensions, batchSize);
            Float16[] halfResult = new Float16[count];
            for (int j = 0; j < count; j++)
            {
              halfResult[j] = (Float16)resultItem.value[j];
            }
            resultArrays.Add((resultItem.name, halfResult));
          }
        }
      }

      return resultArrays;
    }


    /// <summary>
    /// Ends profiling.
    /// </summary>
    public void EndProfiling()
    {
      Session.EndProfiling();
    }


    /// <summary>
    /// Returns a string description of this object.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return "<NetExecutorONNXRuntime " + ONNXFileName + " (" + PrecisionNumBits + ")>";
    }


    /// <summary>
    /// Disposes of this object.
    /// </summary>
    public void Dispose()
    {
      if (!disposed)
      {
        runOptions.Dispose();
        Session.Dispose();
        disposed = true;
      }
    }

  }
}
