using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Modules.Common
{
    /// <author>GoodAI</author>
    /// <meta>jv</meta>
    /// <status>Working</status>
    /// <summary>A node for gating signals from two input branches based on current iteration.</summary>
    /// <description>
    /// If the iteration is lower than the parameter value, Input1 is copied to the Output. 
    /// Interation counter is increased each iteration and is reset at each time step. 
    /// Can be used in the LoopGroup for conditional gating of two signals AND/OR for providing the current iteration number.
    /// 
    /// </description>
    public class MyConditionalGate : MyWorkingNode
    {
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> IterationOutput
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
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

        public override void UpdateMemoryBlocks()
        {
            int c = 1, ch = 1;
            if (Input1 != null)
            {
                c = Input1.Count;
                ch = Input1.ColumnHint;
            }
            else if (Input2 != null)
            {
                c = Input2.Count;
                ch = Input2.ColumnHint;
            }
            Output.Count = c;
            Output.ColumnHint = ch;

            IterationOutput.Count = 1;
            IterationOutput.ColumnHint = 1;
        }

        public override void Validate(MyValidator validator)
        {
            if (ConditionalGateInputs.Enabled && (Input1 == null || Input2 == null))
            {
                validator.AddWarning(this, "One of Inputs is not connected, will publish zeros instead");
                return;
            }
            validator.AssertError(GetInputSize(0) == GetInputSize(1), this, "Input sizes differ!");
        }

        public override string Description
        {
            get
            {
                return "Conditional Gate";
            }
        }

        public MyCountInterationsTask CountIterations{ get; private set; }
        public MyGateTask ConditionalGateInputs { get; private set; }

        private int m_iteration;

        /// <summary>
        /// At each call, the Interation counter is increased by 1. The counter is reset at each change of SimulationStep.
        /// 
        /// <br>Note that use of this gate in multiple nested LoopGroups (and/or any group with custom task planning) can lead to unexpected behavior, 
        /// since the Iteration coutner is reser only on -change- of simulation step.</br>
        /// 
        /// </summary>
        [Description("Count Iterations"), MyTaskInfo(Disabled = false, OneShot = false)]
        public class MyCountInterationsTask : MyTask<MyConditionalGate>
        {
            private uint m_prevSimulationStep;
           
            public override void Init(Int32 nGPU)
            {
                Owner.m_iteration = 0;
                m_prevSimulationStep = uint.MaxValue;
            }

            public override void Execute()
            {
                if (m_prevSimulationStep != SimulationStep)
                {
                    m_prevSimulationStep = SimulationStep;
                    Owner.m_iteration = -1;
                }
                Owner.m_iteration++;

                Owner.IterationOutput.SafeCopyToHost();
                Owner.IterationOutput.Host[0] = Owner.m_iteration;
                Owner.IterationOutput.SafeCopyToDevice();
            }
        }

        /// <summary>
        /// Performs gating based on counter how many times has been called during this simulation step.
        /// If Iteration lower than IterationThreshold, copy Input1 to the Output, Input2 otherwise. Iteration is reset each time step.
        /// Inputs do not have to be connected at all.
        /// </summary>
        [Description("Gate Inputs"), MyTaskInfo(Disabled=false, OneShot=false)]
        public class MyGateTask : MyTask<MyConditionalGate>
        {
            private int m_iterationThreshold = 1;
            [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = 1),
            Description("If the current iteration is lower that the parameter, Input1 is passed to the Output. Input2 otherwise.")]
            public int IterationThreshold
            {
                get
                {
                    return m_iterationThreshold;
                }
                set
                {
                    if (value >= 0)
                    {
                        m_iterationThreshold = value;
                    }
                }
            }

            public override void Init(Int32 nGPU)
            {
            }

            public override void Execute()
            {
                if (Owner.m_iteration < m_iterationThreshold)
                {
                    if (Owner.Input1 == null)
                    {
                        Owner.Output.Fill(0);
                    }
                    else
                    {
                        Owner.Output.CopyFromMemoryBlock(Owner.Input1, 0, 0, Owner.Input1.Count);
                    }
                }
                else
                {
                    if (Owner.Input2 == null)
                    {
                        Owner.Output.Fill(0);
                    }
                    else
                    {
                        Owner.Output.CopyFromMemoryBlock(Owner.Input2, 0, 0, Owner.Input2.Count);
                    }
                }
            }
        }
    }
}
