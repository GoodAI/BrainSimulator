using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using GoodAI.Platform.Core.Nodes;
using GoodAI.TypeMapping;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using YAXLib;

namespace GoodAI.Modules.School.Worlds
{

    public class SchoolWorld : MyWorld, IModelChanger, IMyCustomExecutionPlanner
    {
        #region Constants
        // Constants defining the memory layout of LTStatus information
        private const int NEW_LT_FLAG = 0;
        private const int NEW_LEVEL_FLAG = NEW_LT_FLAG + 1;
        private const int NEW_TU_FLAG = NEW_LEVEL_FLAG + 1;
        private const int LT_IDENTIFIER = NEW_TU_FLAG + 1;
        private const int LEVEL_INDEX = LT_IDENTIFIER + 1;
        private const int TU_INDEX = LEVEL_INDEX + 1;
        private const int LT_STATUS_COUNT = TU_INDEX + 1;
        #endregion

        #region Input and Output MemoryBlocks
        [MyInputBlock]
        public MyMemoryBlock<float> ActionInput
        {
            get { return GetInput(0); }
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Visual
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> Text
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock(2)]
        public MyMemoryBlock<float> Data
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock(3)]
        public MyMemoryBlock<float> DataLength
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        [MyOutputBlock(4)]
        public MyMemoryBlock<float> Reward
        {
            get { return GetOutput(4); }
            set { SetOutput(4, value); }
        }

        // Memory block informing the agent of changes in learning task,
        // level, and training unit.
        //
        // Consists of 
        // - flags signifying a new task, level, and unit, respectively
        // - numbers identifying the current task, level, and unit
        [MyOutputBlock(5)]
        public MyMemoryBlock<float> LTStatus
        {
            get { return GetOutput(5); }
            set { SetOutput(5, value); }
        }



        #endregion

        #region MemoryBlocks sizes
        [MyBrowsable, Category("World Sizes")]
        [YAXSerializableField(DefaultValue = 65536)] // 196608 768 * 256
        public int VisualSize { get; set; }

        [MyBrowsable, Category("World Sizes")]
        [YAXSerializableField(DefaultValue = 1000)]
        public int TextSize { get; set; }

        [MyBrowsable, Category("World Sizes")]
        [YAXSerializableField(DefaultValue = 100)]
        public int DataSize { get; set; }

        public override void UpdateMemoryBlocks()
        {
            Visual.Count = VisualSize;
            Text.Count = TextSize;
            Data.Count = DataSize;
            DataLength.Count = 1;
            Reward.Count = 1;
            LTStatus.Count = LT_STATUS_COUNT;
            CurrentWorld.UpdateMemoryBlocks();
        }
        #endregion

        #region World machinery

        [YAXSerializableField, YAXSerializeAs("CurrentWorld"), YAXCustomSerializer(typeof(WorldAdapterSerializer))]
        private IWorldAdapter m_currentWorld;

        private bool m_switchModel = true;

        [MyBrowsable, Category("World"), TypeConverter(typeof(IWorldAdapterConverter))]
        public IWorldAdapter CurrentWorld
        {
            get
            {
                return m_currentWorld;
            }
            set
            {
                // TODO m_currentWorld Init memory of wrapped world
                m_switchModel = true;
                m_currentWorld = value;
                m_currentWorld.School = this;
                m_currentWorld.UpdateMemoryBlocks();
                m_currentWorld.InitAdapterMemory();
            }
        }

        public override void UpdateAfterDeserialization()
        {
            CurrentWorld = m_currentWorld;
        }
        #endregion

        public SchoolWorld()
        {
            Visual.Metadata[MemoryBlockMetadataKeys.RenderingMethod] = RenderingMethod.Raw.ToString();
        }

        Random m_rndGen = new Random();

        public SchoolCurriculum Curriculum { get; set; }
        ILearningTask m_currentLearningTask;

        // The curriculum to use.
        [MyBrowsable, Category("Curriculum"), Description("Choose which type of curriculum you want to use.")]
        [YAXSerializableField(DefaultValue = CurriculumType.TrainingCurriculum)]
        public CurriculumType TypeOfCurriculum { get; set; }

        // For testing the progression of learning tasks when we don't have an agent or
        // available agents can't complete the task, we can emulate training unit success
        // with the probability set by this parameter.
        [MyBrowsable, Category("World"), Description("Set to 0 < p <= 1 to emulate the success (with probability p) of training units.")]
        [YAXSerializableField(DefaultValue = 0)]
        public float EmulatedUnitSuccessProbability { get; set; }

        // The phases of learning task processing
        enum LTState
        {
            NoLearningTask,     // No learning task -- do nothing
            NewLearningTask,    // New learning task -- initialize
            FirstTU,            // Present first training unit
            TUInProcess,        // Continue processing current unit
            NextTU,             // Present new unit
            AbilityLearned,     // The ability was learned
            AbilityFailed       // The ability failed
        };
            

        public override void Validate(MyValidator validator)
        {
            validator.AssertError(ActionInput != null, this, "ActionInput must not be null");
        }

        public MyNode AffectedNode { get { return this; } }

        public bool ChangeModel(IModelChanges changes)
        {
            if (!m_switchModel)
                return false;

            CurrentWorld.World.EnableDefaultTasks();
            changes.AddNode(CurrentWorld.World);
            m_switchModel = false;
            return true;
        }

        public virtual MyExecutionBlock CreateCustomInitPhasePlan(MyExecutionBlock defaultInitPhasePlan)
        {
            var executionPlanner = TypeMap.GetInstance<IMyExecutionPlanner>();

            MyExecutionBlock plan = executionPlanner.CreateNodeExecutionPlan(CurrentWorld.World, true);

            return new MyExecutionBlock(new IMyExecutable[] { defaultInitPhasePlan, plan });
        }

        public virtual MyExecutionBlock CreateCustomExecutionPlan(MyExecutionBlock defaultPlan)
        {
            var executionPlanner = TypeMap.GetInstance<IMyExecutionPlanner>();

            IMyExecutable[] thisWorldTasks = defaultPlan.Children;

            var blocks = new List<IMyExecutable>();
            // The default plan will only contain one block with: signals in, world tasks, signals out.
            blocks.Add(thisWorldTasks[0]);
            blocks.Add(thisWorldTasks[1]);
            blocks.Add(executionPlanner.CreateNodeExecutionPlan(CurrentWorld.World, false));
            blocks.AddRange(thisWorldTasks.Skip(2));

            return new MyExecutionBlock(blocks.ToArray());
        }

        public void ExecuteLearningTaskStep()
        {
            ResetLTStatusFlags();

            UpdateLTState();

            switch (GetLTState())
            {
                // No learning task -- do nothing
                case LTState.NoLearningTask:
                    return;
                // New learning task -- initialize
                case LTState.NewLearningTask:
                    StartLearningTask();
                    break;
                // Present first training unit
                case LTState.FirstTU:
                    m_currentLearningTask.HandlePresentNewTrainingUnit();
                    NotifyNewTrainingUnit(false);
                    break;
                // Continue processing current unit
                case LTState.TUInProcess:
                    return;
                // Present new unit
                case LTState.NextTU:
                    bool didIncreaseLevel = m_currentLearningTask.HandlePresentNewTrainingUnit();
                    NotifyNewTrainingUnit(didIncreaseLevel);
                    break;
                // The ability was learned
                case LTState.AbilityLearned:
                    m_currentLearningTask = Curriculum.GetNextLearningTask();
                    if (m_currentLearningTask == null)
                    {
                        if (Owner.SimulationHandler.CanPause)
                        {
                            Owner.SimulationHandler.PauseSimulation();
                        }
                    }
                    else
                    { 
                        MyLog.Writer.WriteLine(MyLogLevel.INFO,
                            "Switching to LearningTask: " +
                            m_currentLearningTask.GetType().ToString().Split(new[] { '.' }).Last()
                        );

                        StartLearningTask();
                    }
                    break;
                // The ability failed
                case LTState.AbilityFailed:
                    m_currentLearningTask = null;
                    break;
            }

            LTStatus.SafeCopyToDevice();
        }

        private LTState GetLTState()
        {
            return m_currentLearningTask == null ? LTState.NoLearningTask :
                !m_currentLearningTask.IsInitialized ? LTState.NewLearningTask :
                !m_currentLearningTask.HasPresentedFirstUnit ? LTState.FirstTU :
                m_currentLearningTask.IsAbilityLearned ? LTState.AbilityLearned :
                m_currentLearningTask.DidAbilityFail ? LTState.AbilityFailed :
                m_currentLearningTask.IsTrainingUnitCompleted ? LTState.NextTU :
                LTState.TUInProcess;
        }

        private void UpdateLTState()
        {
            if (m_currentLearningTask != null && m_currentLearningTask.HasPresentedFirstUnit)
            {
                m_currentLearningTask.UpdateState();
            }
        }

        private void StartLearningTask()
        {
            m_currentLearningTask.StartLearningTask();
            NotifyNewLearningTask();
        }

        // Reset the flags signalling new learning task, training unit, or level
        private void ResetLTStatusFlags()
        {
            LTStatus.Host[NEW_LT_FLAG] = 0;
            LTStatus.Host[NEW_TU_FLAG] = 0;
            LTStatus.Host[NEW_LEVEL_FLAG] = 0;
        }

        // Notify of the start of a new curriculum
        private void NotifyNewCurriculum()
        {
            // Will be incremented when LT is presented
            LTStatus.Host[LT_IDENTIFIER] = -1;
        }

        // Notify of the start of a new learning task
        private void NotifyNewLearningTask()
        {
            LTStatus.Host[NEW_LT_FLAG] = 1;
            LTStatus.Host[NEW_LEVEL_FLAG] = 1;
            LTStatus.Host[TU_INDEX] = 0;
            LTStatus.Host[LT_IDENTIFIER] = LTStatus.Host[LT_IDENTIFIER] + 1;
        }

        // Notify of the start of a new training unit and (possibly) level
        private void NotifyNewTrainingUnit(bool didIncreaseLevel)
        {
            LTStatus.Host[NEW_TU_FLAG] = 1;
            if (didIncreaseLevel)
            {
                LTStatus.Host[NEW_LEVEL_FLAG] = 1;
                LTStatus.Host[LEVEL_INDEX] = LTStatus.Host[LEVEL_INDEX] + 1;
            }

            if (LTStatus.Host[NEW_LT_FLAG] == 1 || didIncreaseLevel)
            {
                LTStatus.Host[TU_INDEX] = 0;
            }
            else
            {
                LTStatus.Host[TU_INDEX] = LTStatus.Host[TU_INDEX] + 1;
            }
        }

        public void InitializeCurriculum()
        {
            Curriculum = SchoolCurriculumPlanner.GetCurriculumForWorld(this);
            NotifyNewCurriculum();
        }

        public void ClearWorld()
        {
            CurrentWorld.ClearWorld();
        }

        // Clear world and reapply training set hints
        public void ClearWorld(TrainingSetHints hints)
        {
            ClearWorld();
            SetHints(hints);
        }

        public void SetHints(TrainingSetHints trainingSetHints)
        {
            foreach (var kvp in trainingSetHints)
            {
                CurrentWorld.SetHint(kvp.Key, kvp.Value);
            }
        }

        // Return true if we are emulating the success of training units
        public bool IsEmulatingUnitCompletion()
        {
            return EmulatedUnitSuccessProbability > 0;
        }

        // Emulate the successful completion with a specified probability of the current training unit
        public bool EmulateIsTrainingUnitCompleted(out bool wasUnitSuccessful)
        {
            wasUnitSuccessful = m_rndGen.NextDouble() < EmulatedUnitSuccessProbability;
            return wasUnitSuccessful;
        }

        public InitSchoolWorldTask InitSchool { get; protected set; }
        public InputAdapterTask AdapterInputStep { get; protected set; }
        public OutputAdapterTask AdapterOutputStep { get; protected set; }
        public LearningStepTask LearningStep { get; protected set; }

        /// <summary>
        /// Initialize the world's curriculum
        /// </summary>
        [MyTaskInfo(OneShot = true)]
        public class InitSchoolWorldTask : MyTask<SchoolWorld>
        {
            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                Owner.InitializeCurriculum();
                Owner.m_currentLearningTask = Owner.Curriculum.GetNextLearningTask();
            }
        }

        /// <summary>
        /// Performs Output memory blocks mapping
        /// </summary>
        public class OutputAdapterTask : MyTask<SchoolWorld>
        {
            public override void Init(int nGPU)
            {
                Owner.CurrentWorld.InitWorldOutputs(nGPU);
            }

            public override void Execute()
            {
                Owner.CurrentWorld.MapWorldOutputs();
            }
        }

        /// <summary>
        /// Performs Input memory blocks mapping
        /// </summary>
        public class InputAdapterTask : MyTask<SchoolWorld>
        {
            public override void Init(int nGPU)
            {
                Owner.CurrentWorld.InitWorldInputs(nGPU);
            }

            public override void Execute()
            {
                Owner.CurrentWorld.MapWorldInputs();
            }
        }

        /// <summary>
        /// Update the state of the training task(s)
        /// </summary>
        public class LearningStepTask : MyTask<SchoolWorld>
        {
            public override void Init(int nGPU)
            {

            }

            public override void Execute()
            {
                Owner.ExecuteLearningTaskStep();
            }
        }

    }
}
