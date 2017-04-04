using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;

namespace GoodAI.BrainSimulator.Nodes
{
    /// <summary>
    /// Represents a collection of tasks, all of the same type, gathered from a selection of nodes (again, all of the same type).
    /// See also the <see>NodeSelection</see> class.
    /// </summary>
    public class TaskSelection : IMyTaskBio
    {
        internal TaskSelection(PropertyInfo taskPropInfo, List<MyWorkingNode> nodes)
        {
            if (nodes.Count == 0)
                throw new ArgumentException("Must not be empty", nameof(nodes));

            m_taskPropInfo = taskPropInfo;
            m_nodes = nodes;
        }

        public MyTask Task => TaskSpecimen;

        public IEnumerable<MyTask> EnumerateTasks() => m_nodes.Select(GetCurrentTask);

        public object[] ToObjectArray() => EnumerateTasks().Cast<object>().ToArray();

        #region IMyTaskBio implementation

        public string Name => TaskSpecimen.Name;

        public bool OneShot => TaskSpecimen.OneShot;

        ///<summary>Returns true only if *all* tasks are enabled (3-state value would be more appropriate)</summary>
        public bool Enabled => EnumerateTasks().All(task => task.Enabled);

        /// <summary>Returns tur if *any* of the tasks is Forbidden.</summary>
        public bool Forbidden => EnumerateTasks().Any(task => task.Forbidden);

        /// <summary>Returns tur if *any* of the tasks is DesignTime.</summary>
        public bool DesignTime => EnumerateTasks().Any(task => task.DesignTime);

        public string TaskGroupName => TaskSpecimen.TaskGroupName;

        #endregion

        private MyTask TaskSpecimen => GetCurrentTask(m_nodes.First());

        private readonly PropertyInfo m_taskPropInfo;
        private readonly List<MyWorkingNode> m_nodes;

        private MyTask GetCurrentTask(MyWorkingNode node) => node.GetTaskByPropertyName(m_taskPropInfo.Name);
    }
}