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
        DebuggingCurriculum
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
    }

    public class SchoolCurriculumPlanner
    {
        public static SchoolCurriculum GetCurriculumForWorld(SchoolWorld world)
        {
            SchoolCurriculum curriculum = new SchoolCurriculum();

            switch (world.TypeOfCurriculum)
            {
                case CurriculumType.TrainingCurriculum:
                    curriculum.AddLearningTask(LearningTaskFactory.CreateLearningTask(typeof(LTDetectColor), world), typeof(RoguelikeWorld));
                    curriculum.AddLearningTask(LearningTaskFactory.CreateLearningTask(typeof(LTDetectShape), world), typeof(RoguelikeWorld));
                    break;
                case CurriculumType.DebuggingCurriculum:
                    curriculum.AddLearningTask(LearningTaskFactory.CreateLearningTask(typeof(LTDebugging), world), typeof(RoguelikeWorld));
                    break;
            }

            return curriculum;
        }
    }
}