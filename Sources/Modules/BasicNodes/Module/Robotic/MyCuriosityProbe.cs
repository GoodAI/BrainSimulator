using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;
using GoodAI.Core.Nodes;


namespace GoodAI.Modules.Robotic
{
    /// <summary>Initialises Curiosity Probe.</summary>
    [Description("Init Curiosity Probe"), MyTaskInfo(OneShot = true)]
    public class MyCuriosityProbeInitTask : MyTask<MyCuriosityProbe>
    {
        public override void Init(int nGPU)
        {
        }

        public override void Execute()
        {
            Owner.RealCommands.Fill(0);
            Owner.VirtualCommands.Fill(0);

            Owner.VirtualState.Fill(0);
            Owner.VirtualTarget.Fill(0);
        }
    }

    /// <summary>Updates Curiosity Probe.</summary>
    [Description("Updates Curiosity Probe"), MyTaskInfo(OneShot = true)]
    public class MyCuriosityProbeUpdateTask : MyTask<MyCuriosityProbe>
    {
        public override void Init(int nGPU)
        {
        }

        public override void Execute()
        {
            Owner.RealCommands.Fill(0);
            Owner.VirtualCommands.Fill(0);

            Owner.VirtualState.Fill(0);
            Owner.VirtualTarget.Fill(0);
        }
    }


    /// <author>GoodAI</author>
    /// <tag>#mm</tag>
    /// <status>Testing</status>
    /// <summary>
    ///   Reads robot-state information and performs curiosity probing of robot-control commands. From this info it tries to creates training data for command-to-state mapping.
    /// </summary>
    /// <description>.</description>
    public class MyCuriosityProbe : MyWorkingNode
    {
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> RealCommands
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> VirtualCommands
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock(2)]
        public MyMemoryBlock<float> VirtualState
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock(3)]
        public MyMemoryBlock<float> VirtualTarget
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> RealState
        {
            get { return GetInput(0); }
        }

        /*
        [ReadOnly(false)]
        [YAXSerializableField, YAXElementFor("IO")]
        public override int InputBranches
        {
            get { return base.InputBranches; }
            set
            {
                base.InputBranches = value;
                m_offsets = new int[value];
            }
        }

        [MyBrowsable, YAXSerializableField(DefaultValue = 0), YAXElementFor("IO")]
        public int OutputColHint { get; set; }

        public int[] m_offsets = new int[0];
        */

        public enum MyCuriosityType
        {
            Random,
            Random2
        }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = MyCuriosityType.Random), YAXElementFor("Behavior")]
        public MyCuriosityType CuriosityType { get; set; }
        /*
        public MyMemoryBlock<CUdeviceptr> InputBlocksPointers { get; private set; }
        public MyMemoryBlock<float> Temp { get; private set; }

        public MyInitTask InitMemoryMapping { get; private set; }
        public MyStackInputsTask StackInputs { get; private set; }
        */

        //Tasks
        MyCuriosityProbeInitTask initTask { get; set; }
        MyCuriosityProbeUpdateTask updateTask { get; set; }

        public MyCuriosityProbe()
        {
            //InputBranches = 1;
            //OutputBranches = 4;
        }

        public void CreateTasks()
        {
            initTask = new MyCuriosityProbeInitTask();
            updateTask = new MyCuriosityProbeUpdateTask();
        }

        public override void UpdateMemoryBlocks()
        {
            RealCommands.Count = 3;
            VirtualState.Count = 11;
            VirtualCommands.Count = 12;
            VirtualTarget.Count = 13;
            VirtualTarget.Count = 14;
        }

        public override void Validate(MyValidator validator)
        {
            for (int i = 0; i < InputBranches; i++)
            {
                MyMemoryBlock<float> ai = GetInput(i);

                if (ai == null)
                    validator.AddError(this, string.Format("Missing input {0}.", i));
            }
        }

        public override string Description
        {
            get
            {
                return CuriosityType.ToString();
            }
        }


    }
}
