using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Testing
{
    /// <author>GoodAI</author>
    /// <meta>df</meta>
    /// <status>Working</status>
    /// <summary>Default world implementation. It does not do anything useful. It is here for default compatibility and testing reasons only.</summary>
    /// <description>The node can generate random output of arbitrary size. If <b>PatternCount</b> property is set then the node will generate a seqence of random patterns. If <b>PatternGroups</b> property is set then the node will assign an ascending label to each pattern within the group.</description>
    public class MyTestingWorld : MyWorld
    {
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> RandomPool
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }


        [MyOutputBlock(2)]
        public MyMemoryBlock<float> Label
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("IO")]
        public int OutputSize { get; set; }

        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("IO")]
        public int ColumnHint { get; set; }

        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("IO")]
        public int PatternCount { get; set; }

        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("IO")]
        public int PatternGroups { get; set; }

        public MyCUDAGenerateInputTask GenerateInput { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            Output.Count = OutputSize;
            Output.ColumnHint = ColumnHint;
            RandomPool.Count = PatternCount * OutputSize;
            RandomPool.ColumnHint = ColumnHint;
            Label.Count = PatternCount / PatternGroups;
            Label.ColumnHint = Label.Count;
        }

        /// <summary>
        /// This taks generates next random output. Exposition time and order can be set.
        /// </summary>
        [Description("Generate random inputs")]
        public class MyCUDAGenerateInputTask : MyTask<MyTestingWorld>
        {
            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 1)]
            public int ExpositionTime { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = false)]
            public bool RandomOrder { get; set; }

            private int m_patternIndex = -1;
            private Random rnd = new Random(Guid.NewGuid().GetHashCode());

            public override void Init(Int32 nGPU)
            {

            }

            public override void Execute()
            {
                if (Owner.RandomPool.Count > 0)
                {
                    if (SimulationStep == 0)
                    {
                        MyKernelFactory.Instance.GetRandDevice(Owner).GenerateUniform(Owner.RandomPool.GetDevice(Owner));
                        m_patternIndex = -1;
                        Owner.Label.Fill(0);
                    }

                    if (SimulationStep % ExpositionTime == 0)
                    {

                        if (RandomOrder)
                        {
                            m_patternIndex = (int)(rnd.NextDouble() * Owner.PatternCount);
                        }
                        else
                        {
                            m_patternIndex++;
                            m_patternIndex %= Owner.PatternCount;
                        }

                        //Owner.Label.Fill(0);
                        Array.Clear(Owner.Label.Host, 0, Owner.Label.Count);
                        Owner.Label.Host[m_patternIndex % Owner.PatternGroups] = 1.00f;
                        Owner.Label.SafeCopyToDevice();
                    }

                    Owner.RandomPool.CopyToMemoryBlock(Owner.Output, m_patternIndex * Owner.OutputSize, 0, Owner.OutputSize);
                }
                else
                {
                    MyKernelFactory.Instance.GetRandDevice(Owner).GenerateUniform(Owner.Output.GetDevice(Owner));
                }
            }

            public void ExecuteCPU()
            {
                for (int i = 0; i < Owner.Output.Count; i++)
                {
                    Owner.Output.Host[i] = (float)rnd.NextDouble();
                }
            }
        }
    }

    public class MyTestingWorldWithInput : MyWorld
    {
        [MyOutputBlock]
        public MyMemoryBlock<float> RandomValues
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyInputBlock]
        public MyMemoryBlock<float> SomeControl
        {
            get { return GetInput(0); }
        }

        [MyBrowsable, Category("I/O")]
        [YAXSerializableField, YAXElementFor("IO")]
        public int OutputSize
        {
            get { return RandomValues.Count; }
            set { RandomValues.Count = value; }
        }

        public MyCUDAGenerateInputTask GenerateInput { get; private set; }

        public override void UpdateMemoryBlocks() { }

        [Description("Generate random inputs")]
        public class MyCUDAGenerateInputTask : MyTask<MyTestingWorldWithInput>
        {
            public override void Init(Int32 nGPU)
            {
                // do nothing here
            }

            public override void Execute()
            {
                MyKernelFactory.Instance.GetRandDevice(Owner).GenerateUniform(Owner.RandomValues.GetDevice(Owner));
            }

            public void ExecuteCPU()
            {
                Random rnd = new Random(Guid.NewGuid().GetHashCode());
                for (int i = 0; i < Owner.RandomValues.Count; i++)
                {
                    Owner.RandomValues.Host[i] = (float)rnd.NextDouble();
                }
            }
        }
    }

    public class MyMultipleIOTestingWorld : MyWorld
    {
        [MyOutputBlock]
        public MyMemoryBlock<float> RandomValues
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> MoreRandomValues
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyBrowsable, Category("I/O")]
        [YAXSerializableField, YAXElementFor("IO")]
        public int OutputSize
        {
            get { return RandomValues.Count; }
            set { RandomValues.Count = value; }
        }

        [MyBrowsable, Category("I/O")]
        [YAXSerializableField, YAXElementFor("IO")]
        public int SecondOutputSize
        {
            get { return MoreRandomValues.Count; }
            set { MoreRandomValues.Count = value; }
        }

        public MyCUDAGenerateInputTask GenerateInput { get; private set; }

        public override void UpdateMemoryBlocks() { }

        [Description("Generate random inputs")]
        public class MyCUDAGenerateInputTask : MyTask<MyMultipleIOTestingWorld>
        {
            public override void Init(Int32 nGPU)
            {
                // do nothing here
            }

            public override void Execute()
            {
                MyKernelFactory.Instance.GetRandDevice(Owner).GenerateUniform(Owner.RandomValues.GetDevice(Owner));
                MyKernelFactory.Instance.GetRandDevice(Owner).GenerateUniform(Owner.MoreRandomValues.GetDevice(Owner));
            }

        }
    }
}
