using BrainSimulator.Memory;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using ManagedCuda;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace BrainSimulator.Transforms
{
    [YAXSerializeAs("Accumulator")]
    public class MyAccumulator : MyTransform
    {

        [MyPersistable]
        public MyMemoryBlock<float> DelayedInputs { get; private set; }        

        [MyBrowsable, Category("Params"), Description("Number of time steps to remember")]
        [YAXSerializableField(DefaultValue = 1)]
        public int DelayMemorySize { get; set; }                      
        
        [MyTaskGroup("Mode")]
        public MyShiftDataTask ShiftData { get; private set; }
        [MyTaskGroup("Mode")]
        public MyApproachValueTask ApproachValue { get; private set; }
        [MyTaskGroup("Mode")]
        public MyQuantizedCopyTask CopyInput { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();

            DelayedInputs.Count = DelayMemorySize * InputSize;

            if (Input != null)
            {
                DelayedInputs.ColumnHint = Input.ColumnHint;
            }
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertError(DelayMemorySize > 0, this, "DelayMemorySize must be a positive integer.");
            validator.AssertError(ApproachValue.Factor <= 1 && ApproachValue.Factor > 0, this, "Factor must be greater then 0 and less or equal to 1.");
            validator.AssertError(ApproachValue.Delta >= 0, this, "Delta must be a positive integer or zero");
        }

        public override string Description
        {
            get
            {
                if (ShiftData.Enabled)
                {
                    return ShiftData.Description;
                }
                else if (ApproachValue.Enabled)
                {
                    return ApproachValue.Description;
                }
                else if (CopyInput.Enabled)
                {
                    return CopyInput.Description;
                }
                else
                {
                    return base.Description;
                }
            }
        }

        [Description("Delay")]
        public class MyShiftDataTask : MyTask<MyAccumulator>
        {
            [MyBrowsable, Category("Params"), Description("At simulation start, use the first available data until the memory is filled")]
            [YAXSerializableField(DefaultValue = true)]
            public bool UseFirstInput { get; set; }

            [MyBrowsable, Category("Params"), Description("Initial value for the delayed memory (used when UseFirstInput is not set)")]
            [YAXSerializableField(DefaultValue = 0f)]
            public float InitialValue { get; set; }

            public override void Init(Int32 nGPU)
            {
                
            }

            public override void Execute()
            {
                if (SimulationStep == 0)
                {
                    Owner.DelayedInputs.CopyFromMemoryBlock(Owner.Input, 0, 0, Owner.OutputSize);
                }

                if (SimulationStep < Owner.DelayMemorySize)
                {
                    if (UseFirstInput)
                    {
                        Owner.Output.CopyFromMemoryBlock(Owner.DelayedInputs, 0, 0, Owner.OutputSize);
                    }
                    else
                    {
                        Owner.Output.Fill(InitialValue);
                    }
                }
                else
                {
                    Owner.Output.CopyFromMemoryBlock(Owner.DelayedInputs, (int)(SimulationStep % Owner.DelayMemorySize) * Owner.OutputSize, 0, Owner.OutputSize);
                }

                if (SimulationStep > 0)
                {
                    Owner.DelayedInputs.CopyFromMemoryBlock(Owner.Input, 0, (int)(SimulationStep % Owner.DelayMemorySize) * Owner.OutputSize, Owner.OutputSize);
                }                
            }

            public string Description
            {
                get
                {
                    return "x(t)=x(t-" + Owner.DelayMemorySize + ")";
                }
            }
        }

        [Description("Approach Value")]
        public class MyApproachValueTask : MyTask<MyAccumulator>
        {
            private MyCudaKernel m_kernel;

            public enum SequenceType
            {
                Arithmetic,
                Geometric,
                Momentum
            }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = SequenceType.Geometric)]
            public SequenceType ApproachMethod { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 0.9f)]
            public float Factor { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 0.1f)]
            public float Delta { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 0)]
            public float Target { get; set; }

            public override void Init(Int32 nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "AddAndApproachValueKernel");
            }

            public override void Execute()
            {
                if (SimulationStep == 0)
                {
                    if (ApproachMethod == SequenceType.Momentum)
                    {
                        Owner.Output.CopyFromMemoryBlock(Owner.Input, 0, 0, Owner.InputSize);
                    }
                    else
                    {
                        Owner.Output.Fill(0.0f);
                    }
                }

                m_kernel.SetupExecution(Owner.OutputSize);
                m_kernel.Run(Target, Delta, Factor, (int)ApproachMethod, Owner.Input, Owner.Output, Owner.OutputSize);
            }

            public string Description
            {
                get
                {                                          
                    return ApproachMethod == SequenceType.Momentum ? "f*x(t-1)+(1-f)*x(t)" : "y(x) -> " + Target;
                }
            }
        }

        [Description("Quantized Copy")]
        public class MyQuantizedCopyTask : MyTask<MyAccumulator>
        {
            [MyBrowsable, Category("Params"), Description("Period between input->output copy")]
            [YAXSerializableField(DefaultValue = 10)]
            public int TimePeriod { get; set; }

            public override void Init(Int32 nGPU)
            {

            }

            public override void Execute()
            {
                if (SimulationStep % TimePeriod == 0)
                {
                    Owner.Input.CopyToMemoryBlock(Owner.Output, 0, 0, Owner.InputSize);
                }
            }

            public string Description
            {
                get
                {
                    return "x(t) = x(t % " + TimePeriod + ")";
                }
            }
        }
    }    
}
