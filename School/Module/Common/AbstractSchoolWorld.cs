using GoodAI.Core.Execution;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.School.Common
{
    /// <author>GoodAI</author>
    /// <meta>mp</meta>
    /// <status>WIP</status>
    /// <summary>Base for the school worlds</summary>
    /// <description>
    ///
    /// </description>
    public abstract class AbstractSchoolWorld : MyWorld, IMyCustomExecutionPlanner
    {
        [MyBrowsable, Category("Params"), Description("Choose which type of curriculum you want to use.")]
        [YAXSerializableField(DefaultValue = CurriculumType.TrainingCurriculum)]
        public CurriculumType TypeOfCurriculum { get; set; }

        // For testing the progression of learning tasks when we don't have an agent or
        // available agents can't complete the task, we can emulate training unit success
        // with the probability set by this parameter.
        [MyBrowsable, Category("Params"), Description("Set to 0 < p <= 1 to emulate the success (with probability p) of training units.")]
        [YAXSerializableField(DefaultValue = 0)]
        public float EmulatedUnitSuccessProbability { get; set; }

        SchoolCurriculum m_curriculum;
        ILearningTask m_currentLearningTask;

        // Random number generator instantiated when emulating training unit completion.
        Random m_random;

        public AbstractSchoolWorld()
        {

        }

        public void InitializeCurriculum()
        {
            //m_curriculum = SchoolCurriculumPlanner.GetCurriculumForWorld(this);
        }

        public override void UpdateMemoryBlocks()
        {

        }

        public override void Validate(MyValidator validator)
        {

        }

        public virtual void ClearWorld()
        {
        }

        // Clear world and reapply training set hints
        public virtual void ClearWorld(TrainingSetHints hints)
        {
            ClearWorld();
            SetHints(hints);
        }

        public virtual void SetHints(TrainingSetHints trainingSetHints)
        {
            foreach (var kvp in trainingSetHints)
            {
                SetHint(kvp.Key, kvp.Value);
            }
        }

        public abstract void SetHint(string attr, float value);

        // probably will not be overridden
        public virtual void ExecuteLearningTaskStep()
        {
            //if (m_currentLearningTask == null)
            //    return;

            //if (m_currentLearningTask.HasPresentedFirstUnit)
            //{
            //    m_currentLearningTask.UpdateState();

            //    if (m_currentLearningTask.IsAbilityLearned)
            //    {
            //        //m_currentLearningTask = m_curriculum.GetNextLearningTask();
            //        if (m_currentLearningTask == null)
            //            return;
            //    }
            //    else if (m_currentLearningTask.DidAbilityFail)
            //    {
            //        m_currentLearningTask = null;
            //        return;
            //    }
            //}

            //if (!m_currentLearningTask.HasPresentedFirstUnit || m_currentLearningTask.IsTrainingUnitCompleted)
            //{
            //    m_currentLearningTask.HandlePresentNewTrainingUnit(this);
            //}
        }

        public virtual MyExecutionBlock CreateCustomExecutionPlan(MyExecutionBlock defaultPlan)
        {
            return defaultPlan;
        }

        public virtual MyExecutionBlock CreateCustomInitPhasePlan(MyExecutionBlock defaultInitPhasePlan)
        {
            return defaultInitPhasePlan;
        }

        // Return true if we are emulating the success of training units
        public bool IsEmulatingUnitCompletion()
        {
            return EmulatedUnitSuccessProbability > 0;
        }

        // Emulate the successful completion with a specified probability of the current training unit
        public bool EmulateIsTrainingUnitCompleted(out bool wasUnitSuccessful)
        {
            if (m_random == null)
            {
                m_random = new Random();
            }
            wasUnitSuccessful = m_random.NextDouble() < EmulatedUnitSuccessProbability;
            return true;
        }

        public InitSchoolTask InitSchool { get; protected set; }
        public LearningTaskStepTask LearningTaskStep { get; protected set; }
        //public WorldStepTask WorldStep { get; protected set; }

        /// <summary>
        /// Initialize the world's curriculum
        /// </summary>
        [MyTaskInfo(OneShot = true)]
        public class InitSchoolTask : MyTask<AbstractSchoolWorld>
        {
            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                Owner.InitializeCurriculum();
                //Owner.m_currentLearningTask = Owner.m_curriculum.GetNextLearningTask();
            }
        }

        /// <summary>
        /// Update the state of the training task(s)
        /// </summary>
        public class LearningTaskStepTask : MyTask<AbstractSchoolWorld>
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
