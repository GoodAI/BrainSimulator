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
    /// <summary>Explores command-space and stores it.</summary>
    [Description("Explores command-space and stores it."), MyTaskInfo(OneShot = true)]
    public class MCPExploreTask : MyTask<MyCuriosityProbe>
    {
        public enum MyExplorationType
        {
            Random,
            Systematic
        }

        public struct Pattern
        {
            public uint Time;
            public float[] Command;
            public float[] State;

            public static float[] getData(MyMemoryBlock<float> memBlock)
            {
                memBlock.SafeCopyToHost();

                float[] d = new float[memBlock.Count];
                Buffer.BlockCopy(memBlock.Host, 0, d, 0, memBlock.Count);

                return d;
            }

            public static void putData(float[] source, MyMemoryBlock<float> destination)
            {
                Buffer.BlockCopy(source, 0, destination.Host, 0, destination.Count);
                destination.SafeCopyToDevice();
            }

            public static Pattern Create(uint time, MyMemoryBlock<float> command, MyMemoryBlock<float> state)
            {
                Pattern p;
                p.State = getData(state);
                p.Command = getData(command);
                p.Time = time;

                return p;
            }
            public static Pattern Create(uint time, float[] command, float[] state)
            {
                Pattern p;
                p.State = state;
                p.Command = command;
                p.Time = time;

                return p;
            }
        };

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = MyExplorationType.Random), YAXElementFor("Behavior")]
        public MyExplorationType ExplorationType { get; set; }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 200u), YAXElementFor("Behavior")]
        public uint MaxOneCommandTime { get; set; }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("Behavior")]
        public int RandomSeed { get; set; }

        protected List<Pattern> m_ActualData;
        protected int m_ActTime;
        protected Random m_Rnd;

        public override void Init(int nGPU)
        {
            m_ActualData = new List<Pattern>();
            m_ActTime = 0;
            m_Rnd = new Random(RandomSeed);
        }

        protected bool ShouldTriggerAnother()
        {
            //TODO add faster stop when nothing happens for some time period
            return m_ActTime >= MaxOneCommandTime;
        }

        protected void TriggerNewCommand()
        {
            if(ExplorationType == MyExplorationType.Random)
            {
                for (int i = 0; i < Owner.RealCommands.Count; ++i)
                {
                    Owner.RealCommands.Host[i] = (float)m_Rnd.NextDouble();
                }
                Owner.RealCommands.SafeCopyToDevice();
            }
            else
            {
                //TODO
            }
        }

        public override void Execute()
        {
            //store actual data
            m_ActualData.Add(Pattern.Create(SimulationStep, Owner.RealCommands, Owner.RealState));

            //trigger new command if needed
            if(ShouldTriggerAnother())
            {
                TriggerNewCommand();
            }
        }
    }

    /// <summary>Generates training data.</summary>
    [Description("Generates training data."), MyTaskInfo(OneShot = true)]
    public class MCPGenerateDataTask : MyTask<MyCuriosityProbe>
    {
        public struct TrainingPattern
        {
            public uint Time;
            public float[] Command;
            public float[] State;
            public float[] Target;

            public static TrainingPattern Create(uint time, float[] command, float[] state, float[] target)
            {
                TrainingPattern p;
                p.Time = time;
                p.Command = command;
                p.State = state;
                p.Target = target;

                return p;
            }
        };

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("Behavior")]
        public int RandomSeed { get; set; }
        protected Random m_Rnd;

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Behavior")]
        public uint TargetDelay { get; set; }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Behavior")]
        public uint IgnoredBegining { get; set; }

        protected List<MCPExploreTask.Pattern> m_NewRawData;
        protected List<TrainingPattern> m_TrainingData;

        public void AddRawData(List<MCPExploreTask.Pattern> data)
        {
            if(m_NewRawData != null)
            {
                throw new Exception("repeated addition of raw data to " + this.ToString() + "!");
            }
            m_NewRawData = data;
        }

        public override void Init(int nGPU)
        {
            m_Rnd = new Random(RandomSeed);

            Owner.VirtualCommands.Fill(0);
            Owner.VirtualState.CopyFromMemoryBlock(Owner.RealState, 0, 0, Owner.RealState.Count);
            Owner.VirtualTarget.Fill(0);
        }

        protected void ProcessRawData()
        {
            //IgnoredBegining;
            //TargetDelay;

            if(m_NewRawData != null)
            {
                int size = m_NewRawData.Count;

                for (int i = (int)IgnoredBegining; i + TargetDelay < size; ++i)
                {
                    TrainingPattern p = TrainingPattern.Create((uint)m_TrainingData.Count, m_NewRawData[i].Command, m_NewRawData[i].State, m_NewRawData[i + (int)TargetDelay].State);
                    m_TrainingData.Add(p);
                }

                m_NewRawData = null;
            }
        }

        protected void SelectTrainingPattern()
        {
            TrainingPattern p = m_TrainingData[m_Rnd.Next(m_TrainingData.Count)];
            MCPExploreTask.Pattern.putData(p.State, Owner.VirtualState);
            MCPExploreTask.Pattern.putData(p.Command, Owner.VirtualCommands);
            MCPExploreTask.Pattern.putData(p.Target, Owner.VirtualTarget);
        }

        public override void Execute()
        {
            ProcessRawData();

            SelectTrainingPattern();
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


        //Tasks
        MCPExploreTask exploreTask { get; set; }
        MCPGenerateDataTask generateTask { get; set; }

        public MyCuriosityProbe()
        {
            //InputBranches = 1;
            //OutputBranches = 4;
        }

        public void CreateTasks()
        {
            exploreTask = new MCPExploreTask();
            generateTask = new MCPGenerateDataTask();
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
                return exploreTask.ExplorationType.ToString();
            }
        }


    }
}
