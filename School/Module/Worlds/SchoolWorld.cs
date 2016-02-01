using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Core.Nodes;
using GoodAI.Modules.School.Common;
using GoodAI.Platform.Core.Nodes;
using GoodAI.TypeMapping;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Modules.School.Worlds
{

    public class SchoolWorld : MyWorld, IModelChanger, IMyCustomExecutionPlanner
    {

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
        #endregion

        #region MemoryBlocks sizes
        [MyBrowsable, Category("World Sizes")]
        [YAXSerializableField(DefaultValue = 10000)]
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
        }
        #endregion

        #region World machinery
        // for serialization purposes
        private string CurrentWorldType
        {
            get
            {
                if (CurrentWorld != null)
                {
                    return CurrentWorldType.ToString();
                }
                else
                {
                    return String.Empty;
                }
            }
            set
            {
                if (String.IsNullOrEmpty(value))
                {
                    CurrentWorld = null;
                }
                else
                {
                    CurrentWorld = (IWorldAdapter)Type.GetType(value);
                }
            }
        }


        private IWorldAdapter m_currentWorld;
        private bool m_switchModel = true;

        [MyBrowsable, Category("World")]
        [YAXDontSerialize, TypeConverter(typeof(IWorldAdapterConverter))]
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
                (m_currentWorld as MyWorld).UpdateMemoryBlocks();
                m_currentWorld.InitAdapterMemory(this);
            }
        }

        #endregion

        SchoolCurriculum m_curriculum;
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


        public override void Validate(MyValidator validator)
        {
            validator.AssertError(ActionInput != null, this, "ActionInput must not be null");
        }

        public MyNode AffectedNode { get {return this;} }

        public bool ChangeModel(IModelChanges changes)
        {
            if (!m_switchModel)
                return false;

            changes.AddNode(CurrentWorld as MyWorld);
            m_switchModel = false;
            return true;
        }

        public virtual MyExecutionBlock CreateCustomInitPhasePlan(MyExecutionBlock defaultInitPhasePlan)
        {
            var executionPlanner = TypeMap.GetInstance<IMyExecutionPlanner>();

            MyExecutionBlock plan = executionPlanner.CreateNodeExecutionPlan(CurrentWorld as MyWorld, true);
            plan.Name = (CurrentWorld as MyWorld).Name;

            return new MyExecutionBlock(new IMyExecutable[] { defaultInitPhasePlan, plan } );
        }

        public virtual MyExecutionBlock CreateCustomExecutionPlan(MyExecutionBlock defaultPlan)
        {
            var executionPlanner = TypeMap.GetInstance<IMyExecutionPlanner>();

            IMyExecutable[] thisWorldTasks = defaultPlan.Children;

            var blocks = new List<IMyExecutable>();
            // The default plan will only contain one block with: signals in, world tasks, signals out.
            blocks.Add(thisWorldTasks[0]);
            blocks.Add(executionPlanner.CreateNodeExecutionPlan(CurrentWorld as MyWorld, false));
            blocks.AddRange(thisWorldTasks.Skip(1));

            return new MyExecutionBlock(blocks.ToArray());
        }

        public void ExecuteLearningTaskStep()
        {
            if (m_currentLearningTask == null)
                return;

            if (m_currentLearningTask.HasPresentedFirstUnit)
            {
                m_currentLearningTask.UpdateState();

                if (m_currentLearningTask.IsAbilityLearned)
                {
                    m_currentLearningTask = m_curriculum.GetNextLearningTask();
                    if (m_currentLearningTask == null)
                        return;
                }
                else if (m_currentLearningTask.DidAbilityFail)
                {
                    m_currentLearningTask = null;
                    return;
                }
            }

            if (!m_currentLearningTask.HasPresentedFirstUnit || m_currentLearningTask.IsTrainingUnitCompleted)
            {
                m_currentLearningTask.HandlePresentNewTrainingUnit();
            }
        }

        public void InitializeCurriculum()
        {
            m_curriculum = SchoolCurriculumPlanner.GetCurriculumForWorld(this);
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
            Random rnd = new Random(); // not effective but this method is intended for testing only
            wasUnitSuccessful = rnd.NextDouble() < EmulatedUnitSuccessProbability;
            return true;
        }

        public InitSchoolWorldTask InitSchool { get; protected set; }
        public AdapterTask AdapterStep { get; protected set; }
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
                Owner.m_currentLearningTask = Owner.m_curriculum.GetNextLearningTask();
            }
        }

        /// <summary>
        /// Performs all memory blocks mapping
        /// </summary>
        public class AdapterTask : MyTask<SchoolWorld>
        {
            public override void Init(int nGPU)
            {

            }

            public override void Execute()
            {
                Owner.CurrentWorld.MapWorlds(Owner);
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
