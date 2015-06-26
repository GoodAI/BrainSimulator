using BrainSimulator.Memory;
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
    public class MyGateInput : MyWorkingNode
    {
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input1
        {
            get { return GetInput(0); }            
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> Input2
        {
            get { return GetInput(1); }            
        }
        
        [YAXSerializableField, YAXSerializeAs("Weight")]
        private float m_weight;

        public override void UpdateMemoryBlocks()
        {
            Output.Count = GetInputSize(0);
            Output.ColumnHint = GetInput(0) != null ? GetInput(0).ColumnHint : 1;
        }

        public override void Validate(MyValidator validator) 
        {
            base.Validate(validator);            
            validator.AssertError(GetInputSize(0) == GetInputSize(1), this, "Input sizes differs!");
        }

        public void SetWeight(float value) 
        {
            if (m_weight >= 0 && m_weight <= 1)
            {
                m_weight = value;
            }
        }

        public float GetWeight()
        {
            return m_weight;
        }

        public override string Description
        {
            get
            {
                return "Gate Inputs";
            }
        }
        
        public MyGateTask GateInputs { get; private set; }

        [Description("Gate Inputs")]
        public class MyGateTask : MyTask<MyGateInput>
        {
            private MyCudaKernel m_kernel;

            public override void Init(Int32 nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "Interpolate");
            }

            public override void Execute()
            {
                m_kernel.SetupExecution(Owner.Output.Count);
                m_kernel.Run(Owner.Input1, Owner.Input2, Owner.Output, Owner.m_weight, Owner.Output.Count);                             
            }
        }
    }
}
