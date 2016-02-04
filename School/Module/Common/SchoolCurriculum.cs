using GoodAI.Modules.School.LearningTasks;
using GoodAI.Modules.School.Worlds;
using GoodAI.School.Worlds;
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
        protected List<ILearningTask> TaskOrder = new List<ILearningTask>();
        private IEnumerator<ILearningTask> m_taskEnumerator;
        // The .NET framework does not provide a generic dictionary that preserves
        // insertion order, so we keep (somewhat redundantly) a list of learning tasks
        // to track task ordering and a dictionary to map learning tasks to world types.
        protected Dictionary<ILearningTask, Type> TaskWorldTypes = new Dictionary<ILearningTask, Type>();

        // for foreach usage
        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        public IEnumerator<ILearningTask> GetEnumerator()
        {
            return TaskOrder.GetEnumerator() as IEnumerator<ILearningTask>;
        }

        // for classic usage
        public ILearningTask GetNextLearningTask()
        {
            if (m_taskEnumerator == null)
                m_taskEnumerator = TaskOrder.GetEnumerator();
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
            TaskOrder.Add(task);
        }

        public void AddLearningTask(ILearningTask task, Type worldType)
        {
            // TODO: if tasks are added by a caller in random order, insert the task after tasks that train the required abilities
            TaskOrder.Add(task);
            TaskWorldTypes.Add(task, worldType);
        }

        public Type GetWorldType(ILearningTask task)
        {
            return TaskWorldTypes[task];
        }

        public void AddLearningTask(SchoolWorld world, Type learningTaskType, Type worldType)
        {
            AddLearningTask(LearningTaskFactory.CreateLearningTask(learningTaskType, world), worldType);
        }

        public void AddLearningTask(SchoolWorld world, Type learningTaskType)
        {
            AddLearningTask(LearningTaskFactory.CreateLearningTask(learningTaskType, world), learningTaskType.BaseType.GetGenericArguments()[0]);
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
                    curriculum.AddLearningTask(world, typeof(LTDetectColor));
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
                    curriculum.AddLearningTask(world, typeof(LTDetectColor));
                    curriculum.AddLearningTask(world, typeof(LTClassComposition));
                    curriculum.AddLearningTask(world, typeof(LTDetectShapeColor));
                    curriculum.AddLearningTask(world, typeof(LT1DApproach));
                    curriculum.AddLearningTask(world, typeof(LTActionWCooldown));
                    curriculum.AddLearningTask(world, typeof(LTSimpleSize));
                    curriculum.AddLearningTask(world, typeof(LTSimpleDistance));
                    curriculum.AddLearningTask(world, typeof(LTSimpleAngle));
                    curriculum.AddLearningTask(world, typeof(LTDetectShapeColor));
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
                    //Visual Equivalence
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
                    curriculum.AddLearningTask(world, typeof(LTDebugging));
                    //curriculum.AddLearningTask(LearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.DetectColor, world).GetType());
                    break;
            }

            return curriculum;
        }
    }
}