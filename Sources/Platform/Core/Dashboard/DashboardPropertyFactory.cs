using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Dashboard;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;

namespace GoodAI.Core.Dashboard
{
    public static class DashboardPropertyFactory
    {
        public static DashboardNodeProperty CreateNodeProperty(MyNode node, string propertyName)
        {
            // This is a node property holding a task. Task groups and simple tasks have to be distinguished.
            return new DashboardNodeProperty(node, node.GetType().GetProperty(propertyName));
        }

        public static DashboardTaskProperty CreateTaskProperty(MyTask task, string propertyName)
        {
            if (propertyName == null)
                propertyName = "Enabled";

            return new DashboardTaskProperty(task, task.GetType().GetProperty(propertyName));
        }

        public static DashboardTaskGroupProperty CreateTaskGroupProperty(TaskGroup taskGroup)
        {
            return new DashboardTaskGroupProperty(taskGroup);
        }

        public static DashboardNodePropertyBase CreateProperty(object target, string propertyName)
        {
            var node = target as MyNode;
            if (node != null)
                return CreateNodeProperty(node, propertyName);

            var task = target as MyTask;
            if (task != null)
                return CreateTaskProperty(task, propertyName);

            var taskGroup = target as TaskGroup;
            if (taskGroup != null)
                return CreateTaskGroupProperty(taskGroup);

            return null;
        }
    }
}
