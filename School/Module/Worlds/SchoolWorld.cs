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
using System.Diagnostics;
using System.Linq;
using YAXLib;
using GoodAI.School.Worlds;
using System.Windows.Forms;

namespace GoodAI.Modules.School.Worlds
{
    public class SchoolEventArgs : EventArgs
    {
        public ILearningTask Task { get; private set; }

        public SchoolEventArgs(ILearningTask task)
        {
            Task = task;
        }
    }

    public class SchoolStatus
    {
        public bool m_isNewCurriculum = false;
        public bool m_isNewLT = false;
        public bool m_isNewTU = false;
        public bool m_isNewLevel = false;

        public int m_counterLT = 0;
        public int m_counterLevel = 0;
        public int m_counterTU = 0;
        public int m_counterSuccesses = 0;
    }

    public class SchoolWorld : MyWorld, IModelChanger, IMyCustomExecutionPlanner
    {
        #region Constants
        // Constants defining the memory layout of LTStatus information
        private const int NEW_LT_FLAG = 0;
        private const int NEW_LEVEL_FLAG = NEW_LT_FLAG + 1;
        private const int NEW_TU_FLAG = NEW_LEVEL_FLAG + 1;
        private const int LT_INDEX = NEW_TU_FLAG + 1;
        private const int LEVEL_INDEX = LT_INDEX + 1;
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
        public MyMemoryBlock<float> RewardMB
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

        private float m_aspectRatio;
        private int m_width;
        private int m_height;

        [MyBrowsable, Category("Sizes - Visual"), DisplayName("\tAspectRatio"), ReadOnly(true)]
        [YAXSerializableField(DefaultValue = 1)]
        public float AspectRatio
        {
            get { return m_aspectRatio; }
            set
            {
                m_aspectRatio = value;
                int count = Width * Height;

                // Get sizes that have the same count of pixels but a different aspect ratio -- wh=c & w/h=r
                m_width = (int)Math.Sqrt(m_aspectRatio * count);
                if (Width > 0)
                    m_height = count / Width; // may leave out a few pixels from count due to integer division
            }
        }

        [MyBrowsable, Category("Sizes - Visual"), DisplayName("\tWidth")]
        [YAXSerializableField(DefaultValue = 256)]
        public int Width
        {
            get { return m_width; }
            set
            {
                if (value == 0)
                    return;
                m_width = Math.Max(0, value);
                m_height = (int)(m_width / AspectRatio);
            }
        }

        [MyBrowsable, Category("Sizes - Visual")]
        [YAXSerializableField(DefaultValue = 256)]
        public int Height
        {
            get { return m_height; }
            set
            {
                if (value == 0)
                    return;
                m_height = Math.Max(0, value);
                m_width = (int)(m_height * AspectRatio);
            }
        }

        [MyBrowsable, Category("Sizes - World")]
        [YAXSerializableField(DefaultValue = 1000)]
        public int TextSize { get; set; }

        [MyBrowsable, Category("Sizes - World")]
        [YAXSerializableField(DefaultValue = 100)]
        public int DataSize { get; set; }

        public override void UpdateMemoryBlocks()
        {
            Visual.Dims = new TensorDimensions(Width, Height);
            Text.Count = TextSize;
            Data.Count = DataSize;
            DataLength.Count = 1;
            RewardMB.Count = 1;
            LTStatus.Count = LT_STATUS_COUNT;

            if (CurrentWorld != null)
                CurrentWorld.UpdateMemoryBlocks();
        }

        #endregion

        private IWorldAdapter m_currentWorld;

        [MyBrowsable, Category("World"), TypeConverter(typeof(WorldAdapterConverter)), YAXDontSerialize]
        public IWorldAdapter CurrentWorld
        {
            get
            {
                return m_currentWorld;
            }
            set
            {
                // TODO m_currentWorld Init memory of wrapped world
                m_currentWorld = value;
                m_currentWorld.School = this;
                m_currentWorld.InitAdapterMemory();
            }
        }


        public override void OnSimulationStateChanged(MySimulationHandler.StateEventArgs args)
        {
            // Notify BS that the model has changed -- it will reuse the old model otherwise and won't call inits on CurrentWorld's tasks when run
            if (args.NewState == MySimulationHandler.SimulationState.STOPPED)
            {
                m_shouldShowNewLearningTask = true;
                CurrentLearningTask = null;
                Curriculum.Reset();
            }
        }

        public SchoolWorld()
        {
            Visual.Metadata[MemoryBlockMetadataKeys.RenderingMethod] = RenderingMethod.Raw;
            Visual.Metadata[MemoryBlockMetadataKeys.ShowCoordinates] = true;
        }

        readonly Random m_rndGen = new Random();

        private bool m_shouldShowNewLearningTask = true;
        private bool m_isAfterChangeModelInit = false;
        private bool m_isAfterChangeModelExecute = false;

        private SchoolStatus m_schoolStatus = new SchoolStatus();

        public SchoolCurriculum Curriculum { get; set; }
        public ILearningTask CurrentLearningTask { get; set; }

        public TrainingResult TaskResult { get; private set; }
        private bool m_drawBlackscreen = false;

        public float EmulatedUnitSuccessProbability { get; set; }

        [MyBrowsable, Category("World"), Description("If true, a black screen will be presented for one step after each success.")]
        [YAXSerializableField(DefaultValue = false)]
        public bool ShowBlackscreen { get; set; }

        public event EventHandler<SchoolEventArgs> LearningTaskNew = delegate { };
        public event EventHandler<SchoolEventArgs> TrainingUnitFinished = delegate { };
        public event EventHandler<SchoolEventArgs> TrainingUnitUpdated = delegate { };
        public event EventHandler<SchoolEventArgs> LearningTaskFinished = delegate { };
        public event EventHandler<SchoolEventArgs> LearningTaskLevelFinished = delegate { };
        public event EventHandler<SchoolEventArgs> LearningTaskNewLevel = delegate { };
        public event EventHandler CurriculumStarting = delegate { };
        public event EventHandler<SchoolEventArgs> CurriculumFinished = delegate { };

        public override void Validate(MyValidator validator)
        {
            if (ActionInput == null)
            {
                validator.AssertError(false, this, "ActionInput must not be null");
                MessageBox.Show("The simulation cannot start because no inputs are provided to ActionInput", "Validation Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else if (Curriculum == null || Curriculum.TasksCount == 0)
            {
                validator.AssertError(false, this, "Curriculum must not be empty. Add or enable some learning tasks. Use AI School GUI from menu View->AI School.");
                MessageBox.Show("Curriculum must not be empty. Add or enable at least one learning task.", "Validation Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        public MyNode AffectedNode { get { return this; } }

        public bool ChangeModel(IModelChanges changes)
        {
            if (!m_shouldShowNewLearningTask)
            {
                return false;
            }

            if (CurrentWorld != null)
            {
                changes.RemoveNode(CurrentWorld.World);
            }
            if (Curriculum.IsLast())
            {
                // stop execution
                if (CurrentLearningTask != null)
                {
                    CurriculumFinished(this, new SchoolEventArgs(CurrentLearningTask));
                    LearningTaskFinished(this, new SchoolEventArgs(CurrentLearningTask));
                }
                CurrentLearningTask = null;
                if (Owner.SimulationHandler.CanPause)
                {
                    Owner.SimulationHandler.PauseSimulation();
                }
                return false;
            }
            if (CurrentLearningTask == null)
                CurriculumStarting(this, EventArgs.Empty);

            if (CurrentLearningTask != null)
                LearningTaskFinished(this, new SchoolEventArgs(CurrentLearningTask));
            CurrentLearningTask = Curriculum.GetNext();
            LearningTaskNew(this, new SchoolEventArgs(CurrentLearningTask));

            CurrentWorld = (IWorldAdapter)Owner.CreateNode(CurrentLearningTask.RequiredWorldType);
            CurrentWorld.World.EnableDefaultTasks();
            changes.AddNode(CurrentWorld.World);
            changes.AddNode(this);

            m_shouldShowNewLearningTask = false;
            m_isAfterChangeModelExecute = true;
            m_isAfterChangeModelInit = true;
            return true;
        }

        public virtual MyExecutionBlock CreateCustomInitPhasePlan(MyExecutionBlock defaultInitPhasePlan)
        {
            if (!m_isAfterChangeModelInit)
            {
                // this if is true at the beginning of simulation
                return defaultInitPhasePlan;
            }

            m_isAfterChangeModelInit = false;

            var executionPlanner = TypeMap.GetInstance<IMyExecutionPlanner>();

            MyExecutionBlock plan = executionPlanner.CreateNodeExecutionPlan(CurrentWorld.World, true);

            // add init tasks that initialize the adapter, but not the InitSchool task,
            // which should be run only once at the very beginning
            var blocks = new List<IMyExecutable>();
            blocks.AddRange(defaultInitPhasePlan.Children.Where(x => x != InitSchool));
            MyExecutionBlock initPhasePlanPruned = new MyExecutionBlock(blocks.ToArray());

            return new MyExecutionBlock(initPhasePlanPruned, plan, AdapterInputStep, AdapterOutputStep, LearningStep);
        }

        public virtual MyExecutionBlock CreateCustomExecutionPlan(MyExecutionBlock defaultPlan)
        {
            if (!m_isAfterChangeModelExecute)
            {
                // this if is true at the beginning of simulation
                return defaultPlan;
            }

            m_isAfterChangeModelExecute = false;

            var executionPlanner = TypeMap.GetInstance<IMyExecutionPlanner>();

            IMyExecutable[] thisWorldTasks = defaultPlan.Children;

            var blocks = new List<IMyExecutable>();
            // The default plan will only contain one block with: signals in, world tasks, signals out.
            blocks.Add(thisWorldTasks.First());
            blocks.Add(AdapterInputStep);
            var worldPlan = executionPlanner.CreateNodeExecutionPlan(CurrentWorld.World, false);
            blocks.AddRange(worldPlan.Children.Where(x => x != CurrentWorld.GetWorldRenderTask()));
            blocks.Add(LearningStep);
            blocks.Add(CurrentWorld.GetWorldRenderTask());
            blocks.Add(AdapterOutputStep);
            blocks.Add(thisWorldTasks.Last());

            return new MyExecutionBlock(blocks.ToArray());
        }

        private void ExecuteLearningTaskStep()
        {
            ResetLTStatusFlags();
            Reward = 0.0f; // resets reward signal

            if (ShowBlackscreen && m_drawBlackscreen)
            {
                // Skip task evaluation, a blackscreen will show up this step
                MoveLTStatusToDevice();
                return;
            }

            if (!CurrentLearningTask.IsInitialized)
            {
                InitNewLearningTask();
            }
            else
            {
                // evaluate previus step
                CurrentLearningTask.ExecuteStep();

                // set new level, training unit or step
                // this also partially sets LTStatus
                TaskResult = CurrentLearningTask.EvaluateStep();

                switch (TaskResult)
                {
                    case TrainingResult.TUInProgress:
                        TrainingUnitUpdated(this, new SchoolEventArgs(CurrentLearningTask));
                        break;

                    case TrainingResult.FinishedTU:
                        NotifyNewTrainingUnit();
                        break;

                    case TrainingResult.FailedLT:
                        if (Owner.SimulationHandler.CanPause)
                            Owner.SimulationHandler.PauseSimulation();
                        return;

                    case TrainingResult.FinishedLevel:
                        CurrentLearningTask.IncreaseLevel();
                        if (CurrentLearningTask.CurrentLevel >= CurrentLearningTask.NumberOfLevels)
                        {
                            TaskResult = TrainingResult.FinishedLT;
                            NotifyNewLearningTask();
                        }
                        else
                        {
                            NotifyNewLevel();
                        }
                        break;

                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }

            // set new learning task
            if (m_schoolStatus.m_isNewLT)
            {
                m_shouldShowNewLearningTask = true;
            }
            // if new TU is requested, present new training unit
            else if (m_schoolStatus.m_isNewTU)
            {
                CurrentLearningTask.PresentNewTrainingUnitCommon();
            }

            MoveLTStatusToDevice();
        }

        private bool InitNewLearningTask()
        {
            //m_currentLearningTask = Curriculum.GetNextLearningTask();

            // end of curriculum - there are no more LTs
            if (CurrentLearningTask == null)
            {
                return false;
            }
            // inform user about new LT
            MyLog.Writer.WriteLine(MyLogLevel.INFO, "Switching to LearningTask: " + CurrentLearningTask.GetTypeName());

            CurrentLearningTask.Init();

            NotifyNewTrainingUnit();
            NotifyNewLevel();

            return true;
        }

        // Notify of the start of a new curriculum
        private void NotifyNewCurriculum()
        {
            m_schoolStatus = new SchoolStatus();
        }

        public void ResetLTStatusFlags()
        {
            m_schoolStatus.m_isNewLT = false;
            m_schoolStatus.m_isNewLevel = false;
            m_schoolStatus.m_isNewTU = false;
        }

        public void MoveLTStatusToDevice()
        {
            LTStatus.Host[NEW_LT_FLAG] = (m_schoolStatus.m_isNewLT) ? 1 : 0;
            LTStatus.Host[NEW_LEVEL_FLAG] = (m_schoolStatus.m_isNewLevel) ? 1 : 0;
            LTStatus.Host[NEW_TU_FLAG] = (m_schoolStatus.m_isNewTU) ? 1 : 0;

            LTStatus.Host[LT_INDEX] = m_schoolStatus.m_counterLT;
            LTStatus.Host[TU_INDEX] = m_schoolStatus.m_counterTU;
            LTStatus.Host[LEVEL_INDEX] = m_schoolStatus.m_counterLevel;

            LTStatus.SafeCopyToDevice();
        }

        // Notify of the start of a new learning task
        public void NotifyNewLearningTask()
        {
            m_schoolStatus.m_isNewLT = true;
            m_schoolStatus.m_counterLT++;
            m_schoolStatus.m_counterTU = 0;
            m_schoolStatus.m_counterLevel = 0;
        }

        public void NotifyNewLevel()
        {
            m_schoolStatus.m_isNewLevel = true;
            m_schoolStatus.m_counterLevel++;
            m_schoolStatus.m_counterTU = 0;

            LearningTaskNewLevel(this, new SchoolEventArgs(CurrentLearningTask));
        }

        public void NotifyNewTrainingUnit()
        {
            m_schoolStatus.m_isNewTU = true;
            m_schoolStatus.m_counterTU++;

            TrainingUnitFinished(this, new SchoolEventArgs(CurrentLearningTask));
        }

        public void InitializeCurriculum()
        {
            NotifyNewCurriculum();
        }

        public void ClearWorld()
        {
            CurrentWorld.ClearWorld();
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

        // Emulate the successful completion with a specified probability of the current training unit
        public bool EmulateIsTrainingUnitCompleted()
        {
            return m_rndGen.NextDouble() < EmulatedUnitSuccessProbability;
        }

        public InitSchoolWorldTask InitSchool { get; protected set; }
        public InputAdapterTask AdapterInputStep { get; protected set; }
        public LearningStepTask LearningStep { get; protected set; }
        public OutputAdapterTask AdapterOutputStep { get; protected set; }

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
            }
        }

        /// <summary>
        /// Performs Input memory blocks mapping
        /// </summary>
        public class InputAdapterTask : MyTask<SchoolWorld>
        {
            public override void Init(int nGPU)
            {
                if (Owner.CurrentWorld != null)
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
                if (Owner.CurrentLearningTask != null)
                    Owner.ExecuteLearningTaskStep();
            }
        }

        /// <summary>
        /// Performs Output memory blocks mapping
        /// </summary>
        public class OutputAdapterTask : MyTask<SchoolWorld>
        {
            public override void Init(int nGPU)
            {
                if (Owner.CurrentWorld != null)
                    Owner.CurrentWorld.InitWorldOutputs(nGPU);
            }

            public override void Execute()
            {
                if (Owner.m_drawBlackscreen)
                {
                    Owner.m_drawBlackscreen = false;
                    Owner.Visual.Fill(0);
                    Owner.TaskResult = TrainingResult.TUInProgress;
                    return;
                }

                if (Owner.ShowBlackscreen)
                {
                    switch (Owner.TaskResult)
                    {
                        case TrainingResult.FinishedTU:
                        case TrainingResult.FinishedLevel:
                            // Display a blackscreen as a notification about the agent's success
                            // delay it to the next step -- the learning tasks won't execute next step as well
                            Owner.m_drawBlackscreen = true;
                            break;
                    }
                }

                Owner.CurrentWorld.MapWorldOutputs();
            }
        }

        public int Level
        {
            get
            {
                if (m_schoolStatus != null)
                {
                    return m_schoolStatus.m_counterLevel;
                }
                return 0;
            }
        }

        public float Reward
        {
            get
            {
                if (RewardMB != null && RewardMB.Host != null)
                {
                    return (int)RewardMB.Host[0];
                }
                return 0;
            }
            set
            {
                RewardMB.Host[0] = value;
            }
        }
    }
}
