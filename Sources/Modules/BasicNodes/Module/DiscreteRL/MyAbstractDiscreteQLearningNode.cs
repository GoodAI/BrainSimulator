using GoodAI.BasicNodes.DiscreteRL.Observers;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using YAXLib;


namespace GoodAI.Modules.Harm
{
    /// <author>GoodAI</author>
    /// <meta>jv</meta>
    /// <status>Working</status>
    /// <summary>
    /// Parent of nodes that use discrete QLearning algorithms.
    /// </summary>
    /// <description>
    /// Parent of nodes that use discrete QLearning memory, which can be observed by the MyQMatrixObserver
    /// </description>
    public abstract class MyAbstractDiscreteQLearningNode : MyWorkingNode, IDiscretePolicyObservable
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

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> CurrentStateOutput
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
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

        public abstract void ReadTwoDimensions(ref float[,] values, ref int[,] labelIndexes,
            int XVarIndex, int YVarIndex, bool showRealtimeUtilities = false, int policyNumber = 0);

        public List<String> GetActionLabels()
        {
            List<String> tmp = new List<String>();
            Rds.ActionManager.Actions.ForEach(item => tmp.Add(item.GetLabel()));
            return tmp;
        }
    }
}
