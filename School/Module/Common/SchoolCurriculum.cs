using GoodAI.Modules.School.LearningTasks;
using GoodAI.Modules.School.Worlds;
using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;

namespace GoodAI.Modules.School.Common
{
    /// <summary>
    /// Holds tasks that an agent should be trained with to gain new abilities
    /// </summary>
    public class SchoolCurriculum : IEnumerable<ILearningTask>
    {
        protected List<ILearningTask> Tasks = new List<ILearningTask>();
        private IEnumerator<ILearningTask> m_taskEnumerator;

        // for foreach usage
        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        public IEnumerator<ILearningTask> GetEnumerator()
        {
            return Tasks.GetEnumerator() as IEnumerator<ILearningTask>;
        }

        /// <summary>
        /// Provides next LearningTask if there is any, null otherwise
        /// </summary>
        /// <returns></returns>
        public ILearningTask GetNextLearningTask()
        {
            if (m_taskEnumerator == null)
                m_taskEnumerator = Tasks.GetEnumerator();
            if (m_taskEnumerator.MoveNext())
                return m_taskEnumerator.Current;
            return null;
        }

        public Type GetWorldForNextLT()
        {
            if (m_taskEnumerator == null && Tasks.First() != null)
                return Tasks.First().RequiredWorld;
            else
            {
                int idx = Tasks.IndexOf(m_taskEnumerator.Current);
                if (Tasks.ElementAt(idx + 1) != null)
                    return Tasks.ElementAt(idx + 1).RequiredWorld;
            }

            return null;
        }

        public void ResetLearningProgress()
        {
            if (m_taskEnumerator != null)
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
}