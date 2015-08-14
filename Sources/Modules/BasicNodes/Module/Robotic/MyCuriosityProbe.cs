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
    /// <summary>Initialization of curiosity probe.</summary>
    [Description("Initialization of curiosity probe."), MyTaskInfo(OneShot = true)]
    public class MCPInitTask : MyTask<MyCuriosityProbe>
    {
        public override void Init(int nGPU)
        {
        }

        public override void Execute()
        {
            if (Owner.VirtualCommands != null)
            {
                Owner.VirtualCommands.Fill(0);
            }

            if (Owner.VirtualState != null)
            {
                Owner.VirtualState.CopyFromMemoryBlock(Owner.RealState, 0, 0, Owner.RealState.Count);
            }

            if (Owner.VirtualTarget != null)
            {
                Owner.VirtualTarget.Fill(0);
            }
        }
    }

    /// <summary>Explores command-space and stores it.</summary>
    [Description("Explores command-space and stores it."), MyTaskInfo(OneShot = false)]
    public class MCPExploreTask : MyTask<MyCuriosityProbe>
    {
        public enum MyExplorationType
        {
            Random,
            Systematic
        }

        public struct Pattern
        {
            public int Time;
            public float[] Command;
            public float[] State;

            public float diffState(Pattern p)
            {
                float sum2 = 0.0f;
                for (int i = 0; i < State.Length; ++i)
                {
                    sum2 += (p.State[i] - State[i]) * (p.State[i] - State[i]);
                }

                return (float)Math.Sqrt(sum2);
            }

            public static void MyBlockCopy(float[] source, float[] target, int n)
            {
                //Buffer.BlockCopy(source, 0, target, 0, n);
                for(int i = 0; i < n; ++i)
                {
                    target[i] = source[i];
                }
            }

            public static float[] getData(MyMemoryBlock<float> memBlock)
            {
                memBlock.SafeCopyToHost();

                float[] d = new float[memBlock.Count];
                MyBlockCopy(memBlock.Host, d, memBlock.Count);

                return d;
            }

            public static void putData(float[] source, MyMemoryBlock<float> destination)
            {
                if (destination != null)
                {
                    MyBlockCopy(source, destination.Host, destination.Count);
                    destination.SafeCopyToDevice();
                }
            }

            public static Pattern Create(int time, MyMemoryBlock<float> command, MyMemoryBlock<float> state)
            {
                Pattern p;
                p.State = getData(state);
                p.Command = getData(command);
                p.Time = time;

                return p;
            }
            public static Pattern Create(int time, float[] command, float[] state)
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
        [YAXSerializableField(DefaultValue = 200), YAXElementFor("Behavior")]
        public int MaxOneCommandTime { get; set; }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 5), YAXElementFor("Behavior")]
        public int MinOneCommandTime { get; set; }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 10), YAXElementFor("Behavior")]
        public int MaxNoStateChangeTime { get; set; }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 0.01f), YAXElementFor("Behavior")]
        public float StateIdentityDiff { get; set; }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("Behavior")]
        public int RandomSeed { get; set; }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 0.01f), YAXElementFor("Behavior")]
        public float NoiseLevel { get; set; }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 1.0f), YAXElementFor("Behavior")]
        public float ProbabilityOfCollectingCommand { get; set; }

        protected List<Pattern> m_ActualData;
        protected int m_ActTime;
        protected Random m_Rnd;
        protected int m_NoStateChangeTime;

        public override void Init(int nGPU)
        {
            m_ActualData = new List<Pattern>();
            m_ActTime = MaxOneCommandTime;//to immediately trigger command
            m_Rnd = new Random(RandomSeed);
            m_NoStateChangeTime = 0;
        }

        protected bool ShouldTriggerAnother()
        {
            //TODO add faster stop when nothing happens for some time period
            if(m_ActTime >= MaxOneCommandTime)
            {
                return true;
            }

            if (m_ActTime > MinOneCommandTime && StateIdentityDiff > m_ActualData[m_ActualData.Count - 1].diffState(m_ActualData[m_ActualData.Count - 2]))
            {
                m_NoStateChangeTime += 1;
            }

            if(m_NoStateChangeTime > MaxNoStateChangeTime)
            {
                return true;
            }

            return false;
        }

        protected void TriggerNewCommand()
        {
            if(ExplorationType == MyExplorationType.Random)
            {
                for (int i = 0; i < Owner.RealCommands.Count; ++i)
                {
                    Owner.RealCommands.Host[i] = 1.0f - 2.0f*(float)m_Rnd.NextDouble();
                }
                Owner.RealCommands.SafeCopyToDevice();
            }
            else
            {
                //TODO
            }
        }

        protected void AddNoiseToCommand()
        {
            if (ExplorationType == MyExplorationType.Random)
            {
                if (NoiseLevel > 0.0f)
                {
                    for (int i = 0; i < Owner.RealCommands.Count; ++i)
                    {
                        //TODO: better add normal distribution sample
                        float value = Owner.RealCommands.Host[i] + NoiseLevel * (1.0f - 2.0f * (float)m_Rnd.NextDouble());
                        Owner.RealCommands.Host[i] = Math.Min(1.0f, Math.Max(-1.0f, value));
                    }
                    Owner.RealCommands.SafeCopyToDevice();
                }
            }
            else
            {
                //TODO
            }
        }

        public override void Execute()
        {
            //store actual data
            m_ActualData.Add(Pattern.Create(m_ActTime, Owner.RealCommands, Owner.RealState));

            //trigger new command if needed
            if(ShouldTriggerAnother())
            {
                if (ProbabilityOfCollectingCommand >= m_Rnd.NextDouble())
                {
                    Owner.generateTask.AddRawData(m_ActualData);
                }

                m_ActTime = 0;
                m_NoStateChangeTime = 0;
                m_ActualData.Clear();

                TriggerNewCommand();
            }

            AddNoiseToCommand();
            m_ActTime += 1;
        }
    }

    /// <summary>Generates training data.</summary>
    [Description("Generates training data."), MyTaskInfo(OneShot = false)]
    public class MCPGenerateDataTask : MyTask<MyCuriosityProbe>
    {
        public struct TrainingPatterns
        {
            public int Id;
            public List<float[]> Commands;
            public List<float[]> States;

            public bool GetControllerPattern(int random, out float[] command, out float[] state, out float[] target)
            {
                if(Commands.Count <= 2)
                {
                    command = null;
                    state = null;
                    target = null; 
                    
                    return false;
                }

                int stateId = random % (Commands.Count - 1);
                int targetId = 1 + stateId + random % (Commands.Count - stateId - 1);
                command = Commands[stateId];
                state = States[stateId];
                target = States[targetId];

                return true;
            }

            public bool GetModelPattern(int random, int targetDelay, out float[] command, out float[] state, out float[] target)
            {
                if(Commands.Count <= targetDelay)
                {
                    command = null;
                    state = null;
                    target = null;

                    return false;
                }
                int stateId = random % (Commands.Count - targetDelay);
                int targetId = targetDelay + stateId;
                command = Commands[stateId];
                state = States[stateId];
                target = States[targetId];

                return true;
            }

            public static TrainingPatterns Create(int id, List<float[]> commands, List<float[]> states)
            {
                if (commands.Count != states.Count || commands.Count < 2)
                {
                    throw new Exception("Too few command or states.");
                }

                TrainingPatterns p;
                p.Id = id;
                p.Commands = commands;
                p.States = states;

                return p;
            }
        };

        public enum PatternType
        {
            ForModel,
            ForController
        }
        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = PatternType.ForModel), YAXElementFor("Behavior")]
        public PatternType DataGenerator { get; set; }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("Behavior")]
        public int RandomSeed { get; set; }
        protected Random m_Rnd;

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Behavior")]
        public int ModelTargetDelay { get; set; }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("Behavior")]
        public int IgnoredBegining { get; set; }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 100), YAXElementFor("Behavior")]
        public int MaxHardExamples { get; set; }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 0.1f), YAXElementFor("Behavior")]
        public float HardExamplesFraction { get; set; }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("Behavior")]
        public int MaxTrainingCommands { get; set; }

        public struct OneShotPattern
        {
            public float[] State;
            public float[] Command;
            public float[] Target;

            public void GetPattern(out float[] command, out float[] state, out float[] target)
            {
                command = Command;
                state = State;
                target = Target;
            }

            public static OneShotPattern Create(float[] command, float[] state, float[] target)
            {
                OneShotPattern p;
                p.Command = command;
                p.State = state;
                p.Target = target;
                return p;
            }
        }

        protected List<MCPExploreTask.Pattern> m_NewRawData;
        protected List<TrainingPatterns> m_TrainingData;
        protected SortedDictionary<Tuple<float, int>, OneShotPattern> m_HardData;
        protected bool m_LastGood;
        protected OneShotPattern m_LastPattern;

        public void AddRawData(List<MCPExploreTask.Pattern> data)
        {
            if(m_NewRawData.Count > 0)
            {
                throw new Exception("repeated addition of raw data to " + this.ToString() + "!");
            }
            if (MaxTrainingCommands == 0 || MaxTrainingCommands > m_TrainingData.Count)
            {
                m_NewRawData.AddRange(data);
            }
        }

        public override void Init(int nGPU)
        {
            m_LastGood = false;
            m_Rnd = new Random(RandomSeed);
            m_NewRawData = new List<MCPExploreTask.Pattern>();
            m_TrainingData = new List<TrainingPatterns>();
            m_HardData = new SortedDictionary<Tuple<float, int>, OneShotPattern>();
        }

        protected void ProcessRawData()
        {
            if(m_NewRawData.Count > 1)
            {
                
                int size = m_NewRawData.Count;

                List<float[]> states = new List<float[]>();
                List<float[]> commands = new List<float[]>();

                for (int i = (int)IgnoredBegining; i < size; ++i)
                {
                    states.Add(m_NewRawData[i].State);
                    commands.Add(m_NewRawData[i].Command);
                }

                TrainingPatterns p = TrainingPatterns.Create(m_TrainingData.Count, commands, states);
                m_TrainingData.Add(p);
            }
            m_NewRawData.Clear();
        }

        protected void SelectTrainingPattern()
        {
            m_LastGood = false;
            float[] state = null;
            float[] command = null;
            float[] target = null;

            if (m_TrainingData.Count > 0)
            {
                if (m_Rnd.NextDouble() < HardExamplesFraction && m_HardData.Count > 0)
                {
                    //reuse a pattern with the highest cost
                    OneShotPattern p = m_HardData.Last().Value;
                    Tuple<float, int> k = m_HardData.Last().Key;
                    p.GetPattern(out command, out state, out target);

                    m_HardData.Remove(k);
                    m_LastGood = true;
                }
                else
                {
                    TrainingPatterns p = m_TrainingData[m_Rnd.Next(m_TrainingData.Count)];
                    if (DataGenerator == PatternType.ForModel)
                    {
                        m_LastGood = p.GetModelPattern(m_Rnd.Next(), ModelTargetDelay, out command, out state, out target);
                    }
                    else
                    {
                        m_LastGood = p.GetControllerPattern(m_Rnd.Next(), out command, out state, out target);
                    }
                }
            }

            if (m_LastGood)
            {
                m_LastPattern = OneShotPattern.Create(command, state, target);

                MCPExploreTask.Pattern.putData(state, Owner.VirtualState);
                MCPExploreTask.Pattern.putData(command, Owner.VirtualCommands);
                MCPExploreTask.Pattern.putData(target, Owner.VirtualTarget);
            }
            else
            {
                //TODO: warning terminal output
            }
        }

        protected void ProcessFeedback()
        {
            if(m_LastGood && Owner.LastVirtualCost != null)
            {
                Owner.LastVirtualCost.SafeCopyToHost();

                float cost = Owner.LastVirtualCost.Host[0];

                if (m_HardData.Count < MaxHardExamples || m_HardData.Count > 0 && m_HardData.First().Key.Item1 < cost)
                {
                    //added SimulationStep as second value to make key unique
                    Tuple<float, int> k = Tuple.Create(cost, (int)SimulationStep);

                    m_HardData.Add(k, m_LastPattern);
                }

                //if buffer limit changed or data exceed limit then remove the easiest ones
                while(m_HardData.Count > 0 && m_HardData.Count > MaxHardExamples)
                {
                    m_HardData.Remove(m_HardData.First().Key);
                }
            }

        }

        public override void Execute()
        {
            ProcessRawData();

            ProcessFeedback();

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

        [MyInputBlock(1)]
        public MyMemoryBlock<float> LastVirtualCost
        {
            get { return GetInput(1); }
        }
        //Tasks
        protected MCPInitTask initTask { get; set; }
        protected MCPExploreTask exploreTask { get; set; }
        public MCPGenerateDataTask generateTask { get; set; }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 3), YAXElementFor("Behavior")]
        public int CommandSize
        {
            get { return RealCommands.Count; }
            set { RealCommands.Count = value; }
        }

        public MyCuriosityProbe()
        {
        }

        public void CreateTasks()
        {
            initTask = new MCPInitTask();
            exploreTask = new MCPExploreTask();
            generateTask = new MCPGenerateDataTask();
        }

        public override void UpdateMemoryBlocks()
        {
            if (RealCommands != null)
            {
                VirtualCommands.Count = RealCommands.Count;
            }

            if (RealState != null)
            {
                VirtualState.Count = RealState.Count;
                VirtualTarget.Count = RealState.Count;
            }
        }

        public override void Validate(MyValidator validator)
        {
            MyMemoryBlock<float> ai = GetInput(0);

            if (ai == null)
                validator.AddError(this, string.Format("Missing input {0}.", 0));

            /*for (int i = 0; i < InputBranches; i++)
            {
                MyMemoryBlock<float> ai = GetInput(i);

                if (ai == null)
                    validator.AddError(this, string.Format("Missing input {0}.", i));
            }*/
        }

        public override string Description
        {
            get
            {
                return "Curiosity Probe"; // exploreTask.ExplorationType.ToString();
            }
        }


    }
}
