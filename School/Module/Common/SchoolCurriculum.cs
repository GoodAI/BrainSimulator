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
        protected List<ILearningTask> Tasks = new List<ILearningTask>();
        private IEnumerator<ILearningTask> m_taskEnumerator;
        public string Name { get; set; }

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

        // necessary for deserialization - maybe implement IList or ICollection instead of IEnumerable
        public void Add(ILearningTask task)
        {
            Tasks.Add(task);
        }

        public void AddLearningTask(ILearningTask task, Type worldType)
        {
            // TODO: if tasks are added by a caller in random order, insert the task after tasks that train the required abilities
            Tasks.Add(task);
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
                    curriculum.AddLearningTask(LearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.DetectColor, world), typeof(RoguelikeWorld).GetType());
                    curriculum.AddLearningTask(LearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.DetectShape, world), typeof(RoguelikeWorld).GetType());
                    break;
                case CurriculumType.DebuggingCurriculum:
                    curriculum.AddLearningTask(LearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.DebuggingTask, world), typeof(RoguelikeWorld).GetType());
                    break;
            }

            return curriculum;
        }
    }
}
