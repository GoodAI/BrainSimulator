using System.Collections;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Policy;
using BrainSimulator.Matrix;
using BrainSimulator.Memory;
using BrainSimulator.Nodes;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using BrainSimulator.VSA;
using ManagedCuda;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using YAXLib;

namespace BrainSimulator.VSA.Hashes
{
    ///<author>Martin Milota</author>
    ///<status>WIP</status>
    ///<summary>Constructs an index key for every element of the input. The range of the indices is [0, <seealso cref="MyHashMapperTask.InternalBinCount"/>).
    /// You can optionally change the dimensionality of the input.</summary>
    ///<description></description>
    public class MyHashMapper : MyRandomPool
    {
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input { get { return GetInput(0); } }

        public MyMemoryBlock<float> Temp { get; set; }


        int InputSize { get { return Input != null ? Input.Count : 0; } }

        private Random random;


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

            Temp.Count = InputSize;
            Temp.ColumnHint = ColumnHint;
        }

        #endregion

        #region Buffer generation

        public static float[] GenerateOffsets(Random random, int vectorSize, float range)
        {
            Debug.Assert(random != null, "Missing the random object.");


            var buffer = new float[vectorSize];

            for (int i = 0; i < buffer.Length; i++)
                buffer[i] = (float)random.NextDouble() * 2;

            return buffer;
        }

        float[] GenerateOffsets()
        {
            random = random ?? new Random(Seed);

            return GenerateOffsets(random, InputSize, 2);
        }

        #endregion


        public MyHashMapperTask HashTask { get; private set; }


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

            [MyBrowsable, Category("LSH")]
            [YAXSerializableField(DefaultValue = 1024)]
            public int InternalBinCount { get; set; }

            [MyBrowsable, Category("LSH")]
            [YAXSerializableField(DefaultValue = 1024)]
            public int OutputBinCount { get; set; }

            [MyBrowsable, Category("LSH")]
            [YAXSerializableField(DefaultValue = true)]
            public bool UseOffsets { get; set; }

            [MyBrowsable, Category("LSH")]
            [YAXSerializableField(DefaultValue = true)]
            public bool DoHashing { get; set; }


            private MyCudaKernel _combineVectorsKernel;
            private MyCudaKernel _hashKernel;
            private MyCudaKernel _noHashKernel;

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

                _hashKernel = MyKernelFactory.Instance.Kernel(nGPU, @"VSA\RandomMapper", "GetIndices_ImplicitSeed");
                _hashKernel.SetupExecution(Owner.Output.Count);

                _noHashKernel = MyKernelFactory.Instance.Kernel(nGPU, @"VSA\RandomMapper", "GetIndices_NoHashing");
                _noHashKernel.SetupExecution(Owner.Output.Count);
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
                            DoHashing, InternalBinCount, OutputBinCount != InternalBinCount);
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
                bool doHashMapping, int internalBinCount, bool doScattering = true)
            {
                Debug.Assert(vectorSize > 0, "Invalid vector size");
                Debug.Assert(outputBinCount > 1, "Requires at least 2 output bins");
                Debug.Assert(combineVectorsKernel != null && hashKernel != null, "Missing kernels");


                // Values are in [-1,1] if they were normalized

                if (offsets != null)
                {
                    // Offset to [-1,3]
                    combineVectorsKernel.Run(input, offsets.Value, output,
                        (int)MyJoin.MyJoinOperation.Addition, vectorSize, vectorSize);
                }

                // Modulate to [0,2]
                combineVectorsKernel.Run(output, misc, output, (int)MyJoin.MyJoinOperation.Modulo, vectorSize, 1);
                
                // Transform to integers in [0,BinSize-1]
                combineVectorsKernel.Run(output, misc + sizeof(float), output, (int)MyJoin.MyJoinOperation.Division_int, vectorSize, 1);

                if (!doHashMapping)
                {
                    noHashKernel.Run(output, output, vectorSize, vectorSize, internalBinCount, outputBinCount, seed);
                }
                else if(doScattering)
                {
                    hashKernel.Run(output, output, vectorSize, vectorSize, outputBinCount, seed);
                }
            }
        }
    }
}
