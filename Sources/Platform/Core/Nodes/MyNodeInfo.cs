using GoodAI.Core.Memory;
using GoodAI.Core.Signals;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using YAXLib;

namespace GoodAI.Core.Nodes
{
    public class MyNodeInfo
    {
        private static Dictionary<Type, MyNodeInfo> NODE_INFO = new Dictionary<Type, MyNodeInfo>();

        public MyNodeInfoAttribute Attributes { get; set; }

        public List<PropertyInfo> InitiableProperties { get; private set; }
        
        public List<PropertyInfo> OwnedMemoryBlocks { get; private set; }
        public List<PropertyInfo> InputBlocks { get; private set; }
        public List<PropertyInfo> OutputBlocks { get; private set; }

        public List<PropertyInfo> RegisteredSignals { get; private set; }

        public Dictionary<string, PropertyInfo> KnownTasks { get; private set; }
        public List<PropertyInfo> TaskOrder { get; private set; }

        public Dictionary<string, List<PropertyInfo>> TaskGroups { get; private set; }

        private MyNodeInfo()
        {
            InitiableProperties = new List<PropertyInfo>();

            OwnedMemoryBlocks = new List<PropertyInfo>();
            InputBlocks = new List<PropertyInfo>();
            OutputBlocks = new List<PropertyInfo>();

            RegisteredSignals = new List<PropertyInfo>();

            KnownTasks = new Dictionary<string, PropertyInfo>();            
            TaskOrder = new List<PropertyInfo>();
            TaskGroups = new Dictionary<string, List<PropertyInfo>>();
        }        

        internal static void CollectNodeInfo(Type type)
        {
            if (!NODE_INFO.ContainsKey(type))
            {
                MyNodeInfo nodeInfo = new MyNodeInfo();
                nodeInfo.Attributes = type.GetCustomAttribute<MyNodeInfoAttribute>(true);

                if (nodeInfo.Attributes == null)
                {
                    nodeInfo.Attributes = new MyNodeInfoAttribute();
                }                               

                foreach (PropertyInfo pInfo in type.GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
                {
                    if (typeof(MyTask).IsAssignableFrom(pInfo.PropertyType))
                    {
                        nodeInfo.KnownTasks.Add(pInfo.Name, pInfo);
                        nodeInfo.TaskOrder.Add(pInfo);

                        MyTaskGroupAttribute groupAttr = pInfo.GetCustomAttribute<MyTaskGroupAttribute>(true);

                        if (groupAttr != null)
                        {
                            if (!nodeInfo.TaskGroups.ContainsKey(groupAttr.Name))
                            {
                                nodeInfo.TaskGroups[groupAttr.Name] = new List<PropertyInfo>();
                            }

                            nodeInfo.TaskGroups[groupAttr.Name].Add(pInfo);
                        }
                    }
                    else if (typeof(MyAbstractMemoryBlock).IsAssignableFrom(pInfo.PropertyType))
                    {
                        MyInputBlockAttribute inAttr = pInfo.GetCustomAttribute<MyInputBlockAttribute>(true);
                        if (inAttr != null)
                        {
                            nodeInfo.InputBlocks.Add(pInfo);
                        }
                        else 
                        {
                            if (typeof(MyWorkingNode).IsAssignableFrom(type))
                            {
                                nodeInfo.OwnedMemoryBlocks.Add(pInfo);
                            }

                            if (pInfo.GetCustomAttribute<MyOutputBlockAttribute>(true) != null)
                            {
                                nodeInfo.OutputBlocks.Add(pInfo);
                            }
                        }                        
                    }
                    else if (typeof(MySignal).IsAssignableFrom(pInfo.PropertyType))
                    {
                        nodeInfo.RegisteredSignals.Add(pInfo);
                    }

                    object defaultValue = pInfo.GetAttributeProperty((YAXSerializableFieldAttribute fa) => fa.DefaultValue);

                    if (defaultValue != null)
                    {
                        nodeInfo.InitiableProperties.Add(pInfo);
                    }
                }

                nodeInfo.InputBlocks = new List<PropertyInfo>(
                    nodeInfo.InputBlocks.OrderBy(x => x.GetCustomAttribute<MyInputBlockAttribute>(true).Order));

                nodeInfo.OutputBlocks = new List<PropertyInfo>(
                    nodeInfo.OutputBlocks.OrderBy(x => x.GetCustomAttribute<MyOutputBlockAttribute>(true).Order));

                nodeInfo.TaskOrder = new List<PropertyInfo>(
                    nodeInfo.TaskOrder.OrderBy(x => (x.GetCustomAttribute<MyTaskInfoAttribute>(true) != null ?
                    x.GetCustomAttribute<MyTaskInfoAttribute>(true).Order : 0)));

                NODE_INFO[type] = nodeInfo;
            }
        }

        public static MyNodeInfo Get(Type type)
        {
            return NODE_INFO[type];
        }
    }
}
