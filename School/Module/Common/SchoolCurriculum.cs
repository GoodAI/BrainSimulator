using GoodAI.Modules.School.LearningTasks;
using GoodAI.Modules.School.Worlds;
using System;
using System.Collections;
using System.Collections.Generic;

namespace GoodAI.Modules.School.Common
{
    public enum CurriculumType
    {
        TrainingCurriculum,
        TetrisCurriculum,
        PongCurriculum,
        DebuggingCurriculum,
        AllLTsCurriculum,
    }

    /// <summary>
    /// Holds tasks that an agent should be trained with to gain new abilities
    /// </summary>
    public class SchoolCurriculum : IEnumerable<ILearningTask>
    {
        protected List<ILearningTask> Tasks = new List<ILearningTask>();
        private IEnumerator<ILearningTask> m_taskEnumerator;

        public string Description { get; set; }

        // for foreach usage
        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        public IEnumerator<ILearningTask> GetEnumerator()
        {
            return Tasks.GetEnumerator() as IEnumerator<ILearningTask>;
        }

        // for classic usage
        public ILearningTask GetNextLearningTask()
        {
            if (m_taskEnumerator == null)
                m_taskEnumerator = Tasks.GetEnumerator();
            if (m_taskEnumerator.MoveNext())
                return m_taskEnumerator.Current;
            return null;
        }

        public void ResetLearningProgress()
        {
            m_taskEnumerator.Reset();
        }

        public void Add(ILearningTask task)
        {
            Tasks.Add(task);
        }

        public void Add(SchoolCurriculum curr)
        {
            foreach (ILearningTask task in curr)
                Add(task);
        }

        public void AddLearningTask(ILearningTask task, Type worldType)
        {
            // TODO: if tasks are added by a caller in random order, insert the task after tasks that train the required abilities
            Tasks.Add(task);
        }

        public void AddLearningTask(SchoolWorld world, Type learningTaskType, Type worldType)
        {
            AddLearningTask(LearningTaskFactory.CreateLearningTask(learningTaskType, world), worldType);
        }

        public void AddLearningTask(SchoolWorld world, Type learningTaskType)
        {
            AddLearningTask(LearningTaskFactory.CreateLearningTask(learningTaskType, world), LearningTaskFactory.GetGenericType(learningTaskType));
        }
    }

    public class SchoolCurriculumPlanner
    {
        public static SchoolCurriculum GetCurriculumForWorld(SchoolWorld world)
        {
            SchoolCurriculum curriculum = new SchoolCurriculum();

            switch (world.TypeOfCurriculum)
            {
                case CurriculumType.TrainingCurriculum:
                    curriculum.AddLearningTask(world, typeof(LTCompatibilityMatching));
                    break;

                case CurriculumType.TetrisCurriculum:
                    curriculum.AddLearningTask(world, typeof(LTTetrisTest));
                    break;

                case CurriculumType.PongCurriculum:
                    curriculum.AddLearningTask(world, typeof(LTPongTest));
                    break;

                case CurriculumType.AllLTsCurriculum:
                    curriculum.AddLearningTask(world, typeof(LTDetectWhite));
                    curriculum.AddLearningTask(world, typeof(LTDetectBlackAndWhite));
                    curriculum.AddLearningTask(world, typeof(LTDetectShape));
                    curriculum.AddLearningTask(world, typeof(LTDetectColor));
                    curriculum.AddLearningTask(world, typeof(LTClassComposition));
                    curriculum.AddLearningTask(world, typeof(LTDetectShapeColor));
                    //curriculum.AddLearningTask(world, typeof(LT1DApproach));
                    //curriculum.AddLearningTask(world, typeof(LTActionWCooldown));
                    curriculum.AddLearningTask(world, typeof(LTSimpleSize));
                    curriculum.AddLearningTask(world, typeof(LTSimpleDistance));
                    curriculum.AddLearningTask(world, typeof(LTSimpleAngle));
                    curriculum.AddLearningTask(world, typeof(LTApproach));
                    //Moving Target
                    //Hidden Target
                    //Conditional Target
                    //Noise in Actions
                    curriculum.AddLearningTask(world, typeof(LTObstacles));
                    //Multiple Targets in Sequence
                    //Shape sorting
                    curriculum.AddLearningTask(world, typeof(LTCopyAction));
                    curriculum.AddLearningTask(world, typeof(LTCopySequence));
                    //Count repetititons
                    //Unsupervised Tetris
                    curriculum.AddLearningTask(world, typeof(LTDetectDifference));
                    curriculum.AddLearningTask(world, typeof(LTDetectSimilarity));
                    curriculum.AddLearningTask(world, typeof(LTCompareLayouts));
                    curriculum.AddLearningTask(world, typeof(LTVisualEquivalence));
                    //Compatibility Matching
                    //Rotate and move to fit
                    //2 Back binary test
                    //Tetris
                    //Unsupervised Pong
                    //Prediction
                    //World model
                    //Identity checking
                    //Pong without bricks
                    //Pong with bricks
                    break;
                case CurriculumType.DebuggingCurriculum:
                    curriculum.AddLearningTask(world, typeof(LTDetectShapeColor));
                    //curriculum.AddLearningTask(LearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.DetectColor, world).GetType());
                    break;
            }

            return curriculum;
        }
    }
}