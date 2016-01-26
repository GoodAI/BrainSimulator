using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.VSA.Hashes
{
    /// <author>GoodAI</author>
    /// <meta>mm</meta>
    /// <status>WIP</status>
    /// <summary>A large vector -- memory -- that can be updated by adding a vector of values to specific indices.</summary>
    /// <description></description>
    public class MyHashingMemory : MyWorkingNode
    {
        public enum AddToIndicesOperation
        {
            Add = MyJoin.MyJoinOperation.Addition,
            Subtract = MyJoin.MyJoinOperation.Subtraction,
            Replace = MyJoin.MyJoinOperation.OR,
        }


        [MyInputBlock(0)]
        public MyMemoryBlock<float> Values { get { return GetInput(0); } }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> Indices { get { return GetInput(1); } }


        [MyOutputBlock(0)]
        public MyMemoryBlock<float> MemoryBlock { get { return GetOutput(0); } set { SetOutput(0, value); } }

        public MyMemoryBlock<float> TempBlock { get; set; }



        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 2048)]
        public int MemorySize { get; set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 64)]
        public int MemoryColumnHint { get; set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 64)]
        public int SymbolSize { get; set; }


        #region MyNode overrides

        public override void Validate(MyValidator validator)
        {
            validator.AssertError(MemorySize > 0, this, "Memory can't be empty.");

            if (Values == null || Indices == null)
            {
                validator.AddError(this, "Missing input is suspicious.");
                return;
            }

            validator.AssertError(Values.Count == Indices.Count, this, "The input vector sizes must be the same.");
            validator.AssertError(Values.Count % SymbolSize == 0, this, "The input vector size must be a multiple of SymbolSize.");
        }

        public override string Description { get { return "f(x,h) = MEM + h(x)"; } }

        public override void UpdateMemoryBlocks()
        {
            MemoryBlock.Count = MemorySize;
            MemoryBlock.ColumnHint = MemoryColumnHint;

            TempBlock.Count = 2 * SymbolSize + 2;
        }

        #endregion


        public RandomInitTask RandomizeMemory { get; private set; }
        public MyAddToIndicesTask AddTask { get; private set; }


        /// <summary>
        /// Randomly initializes the contents of the memory.
        /// </summary>
        [Description("Randomize memory contents"), MyTaskInfo(OneShot = true, Disabled = true)]
        public class RandomInitTask : MyTask<MyHashingMemory>
        {
            [MyBrowsable, Category("Structure")]
            [YAXSerializableField(DefaultValue = 0)]
            public float Mean { get; set; }

            [MyBrowsable, Category("Structure")]
            [YAXSerializableField(DefaultValue = 1f / 2048)]
            public float Variance { get; set; }


            public override void Init(int nGPU)
            { }

            public override void Execute()
            {
                if (Owner.MemoryBlock.Host == null)
                    Owner.MemoryBlock.SafeCopyToHost();

                MyRandomPool.GenerateRandomNormalVectors(Owner.MemoryBlock.Host, new Random(Owner.Id), Owner.MemoryBlock.Count, 1, Mean, Variance);
                Owner.MemoryBlock.SafeCopyToDevice();
            }
        }


        /// <summary>
        /// Performs the mapping to the memory.
        /// </summary>
        [Description("Add Values to Indices")]
        public class MyAddToIndicesTask : MyTask<MyHashingMemory>
        {
            private MyCudaKernel _polynomialFuncKernel;
            private MyCudaKernel _combineVectorsKernel;
            private MyCudaKernel _mapToIdcsKernel;
            private MyCudaKernel _constMulKernel;
            private MyProductKernel<float> _dotKernel;


            [MyBrowsable, Category("Structure")]
            [YAXSerializableField(DefaultValue = 1f)]
            public float AddFactor { get; set; }

            [MyBrowsable, Category("Structure")]
            [YAXSerializableField(DefaultValue = 1f)]
            public float DecayFactor { get; set; }


            [MyBrowsable, Category("Mode")]
            [YAXSerializableField(DefaultValue = false)]
            public bool NormalizeTarget { get; set; }

            [MyBrowsable, Category("Mode")]
            [YAXSerializableField(DefaultValue = AddToIndicesOperation.Add)]
            public AddToIndicesOperation AddOperation { get; set; }


            protected MyMemoryBlock<float> Temp { get { return Owner.TempBlock; } }

            protected MyMemoryBlock<float> Memory { get { return Owner.MemoryBlock; } }


            public override void Init(int nGPU)
            {
                if (DecayFactor != 1f)
                {
                    if (DecayFactor > 1f)
                        MyLog.WARNING.WriteLine("Decay factor on a HashingMemoryNode that is greater than one is suspicious...");

                    _polynomialFuncKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "PolynomialFunctionKernel");
                    _polynomialFuncKernel.SetupExecution(Memory.Count);
                }

                if (AddFactor != 1f)
                {
                    _constMulKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "PolynomialFunctionKernel");
                    _constMulKernel.SetupExecution(Owner.SymbolSize);
                }

                if (NormalizeTarget)
                {
                    _combineVectorsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "CombineTwoVectorsKernel");
                    _combineVectorsKernel.SetupExecution(Owner.SymbolSize);
                    _mapToIdcsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "MapToIdcs");
                    _mapToIdcsKernel.SetupExecution(Owner.SymbolSize);
                    _dotKernel = MyKernelFactory.Instance.KernelProduct<float>(Owner, nGPU, ProductMode.f_DotProduct_f);

                }
                else
                {
                    _mapToIdcsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"common\CombineVectorsKernel", "AddToIdcs");
                    _mapToIdcsKernel.SetupExecution(Owner.SymbolSize);
                }

                Temp.SafeCopyToHost();
            }

            public override void Execute()
            {
                if (DecayFactor < 1f)
                    _polynomialFuncKernel.Run(0, 0, DecayFactor, 0, Memory, Memory, Memory.Count);


                var symbolSize = Owner.SymbolSize;

                for (int i = 0; i < Owner.Values.Count / symbolSize; i++)
                {
                    var value = Owner.Values.GetDevicePtr(Owner.GPU, i * symbolSize);
                    var index = Owner.Indices.GetDevicePtr(Owner.GPU, i * symbolSize);
                    var tmp = Temp.GetDevicePtr(Owner, symbolSize);


                    if (NormalizeTarget)
                    {
                        var src = value;

                        if (AddOperation != AddToIndicesOperation.Replace)
                        {
                            _combineVectorsKernel.Run(Memory, index, Temp, (int)MyJoin.MyJoinOperation.Permutation, symbolSize);

                            if (AddFactor != 1)
                            {
                                _constMulKernel.Run(0, 0, AddFactor, 0, value, tmp, symbolSize);
                                src = tmp;
                            }

                            _combineVectorsKernel.Run(src, Temp, Temp, (int)AddOperation, symbolSize);
                            src = Temp.GetDevicePtr(Owner);
                        }

                        //ZXC _dotKernel.Run(Temp, 2 * symbolSize, src, src, symbolSize, /* distributed: */ 0);
                        _dotKernel.outOffset = 2 * symbolSize;
                        _dotKernel.Run(Temp, src, src, symbolSize);

                        _mapToIdcsKernel.Run(src, Temp.GetDevicePtr(Owner.GPU, 2 * symbolSize), index, Memory, symbolSize);
                    }
                    else
                    {
                        var src = value;

                        if (AddFactor != 1)
                        {
                            _constMulKernel.Run(0, 0, AddFactor, 0, value, tmp, symbolSize);
                            src = tmp;
                        }

                        _mapToIdcsKernel.Run(src, index, Memory, (int)AddOperation, symbolSize);
                    }
                }
            }
        }
    }
}
