using BrainSimulator;
using BrainSimulator.Nodes;
using BrainSimulator.Memory;
using BrainSimulator.Utils;
using BrainSimulator.Task;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using System.Threading.Tasks;
using YAXLib;
using ManagedCuda;

namespace BrainSimulator.Harm
{
    /// <author>Jaroslav Vitku</author>
    /// <status>Working</status>
    /// <summary>Implements motivation-based action selection method. Motivation weights between greedy and random strategy.</summary>
    /// <description>
    /// 
    /// <ul>
    ///     <li> Uses epsilon-greedy action seleciton for selecting action based on utility values on inputs.</li>
    ///     <li> Epsilon defines amount of randomization in the greedy strategy and is altered by the motivation input: the higher motivation -> the less randomization (more greedy strategy).</li>
    ///     <li> Selected action is published in 1ofN code.</li>
    /// </ul>
    /// 
    /// <h3>Inputs</h3>
    /// <ul>
    /// <li><b>Utilities: </b>Vector of utility values corresponding to particular actions (e.g. produced by the DiscreteQLearningNode).</li>
    /// <li><b>Motivation: </b>Current amount of motivation to use the learned strategy (weights exploitation vs. exploration of the strategy).</li>
    /// </ul>
    /// <h3>Outputs</h3>
    /// <ul>
    /// <li><b>SelectedAction: </b>Action that was selected, coded in 1ofN code (e.g. to be received by the DiscreteQLearningNode(s)).</li>
    /// </ul>
    /// </description>
    public class MyActionSelectionNode : MyWorkingNode
    {
        [MyInputBlock]
        public MyMemoryBlock<float> UtilityInput
        {
            get { return GetInput(0); }
        }

        [MyInputBlock]
        public MyMemoryBlock<float> MotivationInput
        {
            get { return GetInput(1); }
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        public MyMemoryBlock<int> MaxUtilInd { get; private set; }

        public MySimpleSortTask Sort { get; private set; }

        public MyActionSelectionNode() { }

        public override void UpdateMemoryBlocks()
        {
            if (UtilityInput != null)
            {
                Output.Count = UtilityInput.Count;
            }
            MaxUtilInd.Count = 1;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (MotivationInput.Count != 1)
            {
                validator.AddError(this, "MotivationInput requires size of 1");
            }
        }
    }

    /// <summary>
    /// Choose the best action according to the probability of M, otherwise completely random.
    /// 
    /// If there are multiple identical highest values, choose randomly from all.
    /// 
    /// <h3>Parameters</h3>
    /// <ul>
    /// <li><b>Selection Period: </b>Select new action each N steps</li>
    /// <li><b>Min Epsilon: </b>Minimum probability of randomization (in case that Motivation=1)</li>
    /// </ul>
    /// </summary>
    [Description("Action Selection"), MyTaskInfo(OneShot = false)]
    public class MySimpleSortTask : MyTask<MyActionSelectionNode>
    {
        [MyBrowsable, Category("Randomization"),
        Description("Minimum probability of randomization (in case that Motivation=1)")]

        [YAXSerializableField(DefaultValue = 0.1f)]
        public float MinEpsilon { get; set; }

        [MyBrowsable, DisplayName("Selection Period"), Category("Action Selection Period"),
        Description("Select new action each N steps")]
        [YAXSerializableField(DefaultValue = 1)]
        public int ASMPeriod { get; set; }

        public MySimpleSortTask() { }

        private MyCudaKernel m_kernel;
        private MyCudaKernel m_setKernel;
        private Random m_rnd;

        public override void Init(int nGPU)
        {
            m_rnd = new Random(DateTime.Now.Millisecond);
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Harm\MatrixQLearningKernel", "findMaxIndMultipleDetector");
            m_setKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Harm\MatrixQLearningKernel", "oneOfNSelection");
        }

        public override void Execute()
        {
            if (this.SimulationStep % ASMPeriod != 0)
            {
                return;
            }

            m_kernel.SetupExecution(1);
            m_setKernel.SetupExecution(Owner.UtilityInput.Count);

            m_kernel.Run(Owner.UtilityInput, Owner.MaxUtilInd, Owner.UtilityInput.Count);

            // MaxInd = -1 means that all utilities were the same => generate index randomly
            Owner.MaxUtilInd.SafeCopyToHost();
            Owner.MotivationInput.SafeCopyToHost();

            // compute epsilon
            float motivation = Owner.MotivationInput.Host[0];
            float epsilon = 1 - motivation;

            if (Owner.MaxUtilInd.Host[0] < 0)   // multiple identical utilities => random
            {
                epsilon = 1;
            }
            else if (epsilon < MinEpsilon)      // min. randomization
            {
                epsilon = MinEpsilon;
            }

            if (m_rnd.NextDouble() <= epsilon)
            {
                Owner.MaxUtilInd.Host[0] = m_rnd.Next(Owner.UtilityInput.Count);
            }
            Owner.MaxUtilInd.SafeCopyToDevice();
            m_setKernel.Run(Owner.Output, Owner.MaxUtilInd, Owner.UtilityInput.Count);
        }
    }       
}
