using BrainSimulator.Memory;
using BrainSimulator.Nodes;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace BrainSimulator.Nodes
{
    public abstract class MyWorld : MyWorkingNode
    {
        public virtual void Cleanup() 
        {
        
        }

        public virtual void DoPause()
        {

        }

        public void ValidateWorld(MyValidator validator)
        {
            ValidateMandatory(validator);
            Validate(validator);
        }

        public override void ProcessOutgoingSignals()
        {
            OutgoingSignals = 0;

            OutgoingSignals |= RiseSignalMask;
            OutgoingSignals &= ~DropSignalMask;
        }
    }
}

namespace BrainSimulator.Testing
{
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

            if (PatternCount > 0)
            {
                RandomPool.Count = PatternCount * OutputSize;
                RandomPool.ColumnHint = ColumnHint;
                Label.Count = PatternCount / PatternGroups;
                Label.ColumnHint = Label.Count;
            }
        }

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

                    if (SimulationStep % ExpositionTime == 0) {

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
