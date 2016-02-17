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
        public ILearningTask GetNext()
        {
            if (m_taskEnumerator == null)
            {
                m_taskEnumerator = Tasks.GetEnumerator();
                m_taskEnumerator.Reset();
            }
            if (m_taskEnumerator.MoveNext())
                return m_taskEnumerator.Current;
            return null;
        }

        public bool IsLast()
        {
            if (m_taskEnumerator == null)
            {
                return (Tasks.Count == 0);
            }
            return Tasks.LastIndexOf(m_taskEnumerator.Current) == Tasks.Count - 1;
        }

        public void Reset()
        {
            m_taskEnumerator = null;
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
    }
}