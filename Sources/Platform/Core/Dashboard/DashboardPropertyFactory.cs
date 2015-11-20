using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Dashboard;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;

namespace GoodAI.Platform.Core.Dashboard
{
    public static class DashboardPropertyFactory
    {
        public static DashboardProperty Create(object target, string propertyName)
        {
            DashboardNodeProperty property = null;

            var node = target as MyNode;
            if (node != null)
            {
                property = new DashboardNodeProperty
                {
                    Node = node,
                    PropertyInfo = node.GetType().GetProperty(propertyName)
                };
            }
            else
            {
                var task = target as MyTask;
                if (task != null)
                {
                    property = new DashboardTaskProperty
                    {
                        Node = task.GenericOwner,
                        Task = task,
                        PropertyInfo = task.GetType().GetProperty(propertyName)
                    };
                }
            }

            if (property == null)
                throw new InvalidOperationException("Invalid property owner provided");

            return property;
        }
    }
}
