using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda.BasicTypes;
using System;
using System.ComponentModel;
using System.Diagnostics;
using ManagedCuda;
using YAXLib;

namespace GoodAI.Modules.VSA.Hashes
{
    /// <author>GoodAI</author>
    /// <meta>mm</meta>
    /// <status>working</status>
    /// <summary>Constructs an index key for every element of the input. The range of the indices is [0, OutputBinCount).</summary>
    /// <description>
    ///
    /// <h3> Features: </h3>
    /// Transforms each input value to an index depending on the mode:
    /// <ol>
    ///     <li><h4>Simple:</h4></li>
    ///     <ul>
    ///         <li>Determinalistically randomizes bits in the output via <a href="https://code.google.com/p/smhasher/wiki/MurmurHash3">MurmurHash3</a>.</li>
    ///         <li>Modulates the output to the integer interval <code>[0, <i>OutputBinCount</i>)</code>.</li>
    ///     </ul>
    /// 
    ///     <li><h4>Locality-sensitive</h4> (due to Datar-Immorlica-Indyk-Mirrokni’04):</li>
    ///     <ul>
    ///         <li>Inputs should floating point numbers from <code>[-1, 1]</code>.</li>
    ///         <li>If <i>UseOffsets</i> is true, adds precomputed random values from <code>[0, 2]</code> to the corresponding inputs.</li>
    ///         <li>Modulates the results by 2.</li>
    ///         <li>Divides by <code>2/<i>InternalBinCount</i></code> and truncates to get the integer index of the bin.</li>
    ///         <li>If <i>DoHashing</i> is true, hashes these values to <code>[0, <i>OutputBinCount</i>)</code> via Simple hashing.</li>
    ///         <li>If <i>DoHashing</i> is false, hashes values to <code>[0, <i>InternalBinCount</i>)</code> and offsets them by <code>i*<i>InternalBinCount</i></code>, where <code>i</code> is the value's index in the vector.
    ///             WARNING: This results in output index range in <code>[0, <i>Input.Count</i>*<i>InternalBinCount</i>)</code>.</li>
    ///     </ul>
    /// </ol>
    ///  
    /// <h3> Important notices: </h3>
    /// <ul>
    ///     <li>OutputBinCount must be a power of 2 for now.</li>
    ///     <li>Consider using the RandomMapper node to change the dimensionality of the inputs (can be handy for more benevolent locality-sensitive hashing).</li>
    ///     <li>Use the RandomMapper node first if you want the inputs' values to be connected -- slightly changing a single value will then affect all the output indices.</li>
    /// </ul>
    /// </description>
    public class MyHashMapper : MyRandomPool
    {
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input { get { return GetInput(0); } }

        public MyMemoryBlock<float> Temp { get; set; }


        int InputSize { get { return Input != null ? Input.Count : 0; } }


        #region MyRandomPool overrides

        protected override string GlobalVariableName { get { return "HASH_MAPPER_" + InputSize + "offsets"; } }

        protected override int PatternCount { get { return InputSize; } }

        public override int Seed { get { return 12345; } }

        #endregion

        #region MyNode overrides

        public override string Description { get { return "f(x) = h(x)"; } }

        public override void UpdateMemoryBlocks()
        {
            Output.Count = InputSize;
            Output.ColumnHint = ColumnHint;

            Temp.Count = Math.Max(2, InputSize);
            Temp.ColumnHint = ColumnHint;
        }

        #endregion

        #region Buffer generation

        float[] GenerateOffsets()
        {
            var random = new Random(Seed);
            var buffer = new float[InputSize];

            for (int i = 0; i < buffer.Length; i++)
                buffer[i] = (float)random.NextDouble() * 2;

            return buffer;
        }

        #endregion


        public MyHashMapperTask HashTask { get; private set; }


        /// <summary>
        /// Performs the hash mapping.
        /// </summary>
        [Description("Hash Input to indices")]
        public class MyHashMapperTask : MyTask<MyHashMapper>
        {
            public enum HashMapperMode
            {
                Simple,
                LocalitySensitive,
            }


            [MyBrowsable, Category("General")]
            [YAXSerializableField(DefaultValue = HashMapperMode.LocalitySensitive)]
            public HashMapperMode Mode { get; set; }

            [MyBrowsable, Category("General"), Description("WARNING: Must be a power of 2. The output indices will all fall in the interval [0, OutputBinCount).")]
            [YAXSerializableField(DefaultValue = 1024)]
            public int OutputBinCount { get; set; }


            [MyBrowsable, Category("LSH"),
            Description("The number of bins to which the input interval gets split." +
                        "The values are scrambled to [0, OutputBinCount) after being assigned to a bin.")]
            [YAXSerializableField(DefaultValue = 1024)]
            public int InternalBinCount { get; set; }

            [MyBrowsable, Category("LSH"), Description("Use random offset for each value before computing the target bin.")]
            [YAXSerializableField(DefaultValue = true)]
            public bool UseOffsets { get; set; }

            [MyBrowsable, Category("LSH"),
            Description("WARNING: If false, outputs indices in the interval [0, Input.Count * InternalBinCount)!." +
                        "If false, designates disjunctive index intervals for every input element instead of random scrambling to [0, OutputBinCount)." +
                        "This allows us to see the LSH properties more clearly.")]
            [YAXSerializableField(DefaultValue = true)]
            public bool DoHashing { get; set; }


            private MyCudaKernel _combineVectorsKernel;
            private MyCudaKernel _hashKernel;
            private MyCudaKernel _noHashKernel;

            CudaStream m_stream;

            public override void Init(int nGPU)
            {
                switch (Mode)
                {
                    case HashMapperMode.Simple:
                        break;

                    case HashMapperMode.LocalitySensitive:
                        MyMemoryManager.Instance.ClearGlobalVariable(Owner.GlobalVariableName, Owner.GPU);

                        // Only values are the modulo and and integer divisor (placing into bins)
                        Owner.Temp.SafeCopyToHost(0, 2);
                        Owner.Temp.Host[0] = 2f;
                        Owner.Temp.Host[1] = 2f / InternalBinCount;
                        Owner.Temp.SafeCopyToDevice(0, 2);

                        break;

                    default:
                        throw new ArgumentOutOfRangeException();
                }
                _combineVectorsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"common\CombineVectorsKernel", "CombineTwoVectorsKernelVarSize");
                _combineVectorsKernel.SetupExecution(Owner.InputSize);

                _hashKernel = MyKernelFactory.Instance.Kernel(nGPU, @"VSA\Mappers", "GetIndices_ImplicitSeed");
                _hashKernel.SetupExecution(Owner.Output.Count);

                _noHashKernel = MyKernelFactory.Instance.Kernel(nGPU, @"VSA\Mappers", "GetIndices_NoHashing");
                _noHashKernel.SetupExecution(Owner.Output.Count);

                m_stream = new CudaStream();
            }

            public override void Execute()
            {
                switch (Mode)
                {
                    case HashMapperMode.Simple:
                        _hashKernel.Run(Owner.Input, Owner.Output, Owner.Input.Count, Owner.Output.Count, OutputBinCount, Owner.Seed);
                        break;

                    case HashMapperMode.LocalitySensitive:
                        CUdeviceptr? offsets = null;

                        if (UseOffsets)
                            offsets = MyMemoryManager.Instance.GetGlobalVariable(Owner.GlobalVariableName, Owner.GPU, Owner.GenerateOffsets).DevicePointer;

                        GetIndices(
                            Owner.Input.GetDevicePtr(Owner.GPU), Owner.Output.GetDevicePtr(Owner.GPU), Owner.Temp.GetDevicePtr(Owner.GPU), offsets,
                            Owner.InputSize, OutputBinCount, Owner.Seed,
                            _combineVectorsKernel, _hashKernel, _noHashKernel,
                            DoHashing, InternalBinCount, m_stream);
                        Owner.Output.SafeCopyToHost();
                        break;

                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }

            /// <summary>
            /// Transforms the <paramref name="output"/> vector into a vector of indices with properties specified by the parameters.
            /// </summary>
            /// <param name="input">The vector to transform.</param>
            /// <param name="output">The memory to contain the results.</param>
            /// <param name="misc">A vector containing the range to modulate to as the first value (typically 2f because dot product ranges from [-1,1])
            /// and the bin size in this modulated space (typically <paramref name="misc"/>[0] / internalBinCount) as the second value.</param>
            /// <param name="offsets">The random offsets for each <paramref name="output"/> value (typically uniform random numbers in [0, <paramref name="misc"/>[0].</param>
            /// <param name="vectorSize">The length of the <paramref name="output"/> vector.</param>
            /// <param name="outputBinCount">The range into which the internal bins will be scattered.</param>
            /// <param name="seed">The seed used for the scattering the internal bins.</param>
            /// <param name="combineVectorsKernel">The kernel used for addition, modulo and integer division.</param>
            /// <param name="hashKernel">The kernel used for scattering the internal bins.</param>
            /// <param name="doScattering">If true, each internal bin will be randomly scattered to an integer in [0, <paramref name="outputBinCount"/>).
            /// Otherwise, the range of the output indices will be [0, internalBinCount)</param>
            public static void GetIndices(
                CUdeviceptr input, CUdeviceptr output, CUdeviceptr misc, CUdeviceptr? offsets,
                int vectorSize, int outputBinCount, int seed,
                MyCudaKernel combineVectorsKernel, MyCudaKernel hashKernel, MyCudaKernel noHashKernel,
                bool doHashMapping, int internalBinCount,
                CudaStream stream)
            {
                Debug.Assert(vectorSize > 0, "Invalid vector size");
                Debug.Assert(outputBinCount > 1, "Requires at least 2 output bins");
                Debug.Assert(combineVectorsKernel != null && hashKernel != null, "Missing kernels");


                // Values are in [-1, 1] if they were normalized

                if (offsets != null)
                {
                    // Offset to [-1, 3]
                    combineVectorsKernel.RunAsync(stream, input, offsets.Value, output, (int)MyJoin.MyJoinOperation.Addition, vectorSize, vectorSize);
                }

                // Modulate to [0, 2]
                combineVectorsKernel.RunAsync(stream, output, misc, output, (int)MyJoin.MyJoinOperation.Modulo, vectorSize, 1);

                // Transform to integers in [0, InternalBinCount - 1]
                combineVectorsKernel.RunAsync(stream, output, misc + sizeof(float), output, (int)MyJoin.MyJoinOperation.Division_int, vectorSize, 1);

                if (doHashMapping)
                {
                    hashKernel.RunAsync(stream, output, output, vectorSize, vectorSize, outputBinCount, seed);
                }
                else
                {
                    noHashKernel.RunAsync(stream, output, output, vectorSize, internalBinCount);
                }
            }
        }
    }
}
