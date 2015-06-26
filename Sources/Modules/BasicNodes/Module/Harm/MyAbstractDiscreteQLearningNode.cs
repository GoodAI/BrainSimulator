using BrainSimulator.Nodes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using BrainSimulator.Memory;
using BrainSimulator.Task;
using BrainSimulator.Transforms;
using BrainSimulator.Utils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System.ComponentModel;
using YAXLib;
using BrainSimulator.Observers;
using BrainSimulator.GridWorld;


namespace BrainSimulator.Harm
{
    /// <author>Jaroslav Vitku</author>
    /// <status>Under development</status>
    /// <summary>
    /// Parent of nodes that use discrete QLearning algorithms.
    /// </summary>
    /// <description>
    /// Parent of nodes that use discrete QLearning memory, which can be observed by the MyQMatrixObserver
    /// </description>
    public abstract class MyAbstractDiscreteQLearningNode : MyWorkingNode
    {
        [MyInputBlock(0)]
        public MyMemoryBlock<float> GlobalDataInput
        {
            get { return GetInput(0); }
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> SelectedActionInput
        {
            get { return GetInput(1); }
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> UtilityOutput
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyBrowsable, Category("IO"), DisplayName("Number of Primitive Actions"),
        Description("Number of primitive actions produced by the agent (e.g. 6 for the current gridworld, 3 for the breakout game)")]
        [YAXSerializableField(DefaultValue = 6)]
        public int NoActions { get; set; }

        [MyBrowsable, Category("IO"), DisplayName("Input Rescale Size"),
        Description("Memory is indexed by integers, user should ideally rescale variable values to fit into integers.")]
        [YAXSerializableField(DefaultValue = 9)]
        public int RescaleVariables { get; set; }

        public MyRootDecisionSpace Rds { get; set; }

        public override void UpdateMemoryBlocks()
        {
        }
    }
}
