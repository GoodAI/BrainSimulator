using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using YAXLib;

namespace GoodAI.Core.Dashboard
{
    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class Dashboard : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;

        [YAXSerializableField]
        public IList<DashboardProperty> Properties { get; set; }

        public Dashboard()
        {
            Properties = new List<DashboardProperty>();
        }

        public bool Contains(object node, string propertyName)
        {
            return Properties.Any(property => property.Equals(node, propertyName));
        }

        public void RestoreFromIds(MyProject project)
        {
            foreach (var property in Properties.ToList())
            {
                try
                {
                    property.RestoreFromId(project);
                }
                catch
                {
                    Properties.Remove(property);
                    MyLog.WARNING.WriteLine("A property with identifier \"{0}\" could not be deserialized.",
                        property.PropertyId);
                }
            }
        }

        public void Add(object target, string propertyName)
        {
            if (Contains(target, propertyName))
                return;

            DashboardProperty property = null;

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

            Properties.Add(property);
            OnPropertiesChanged("Properties");
        }

        public void Remove(object target, string propertyName)
        {
            var property = Properties.FirstOrDefault(p => p.Equals(target, propertyName));
            if (property != null && Properties.Remove(property))
            {
                OnPropertiesChanged("Properties");
            }
        }

        protected virtual void OnPropertiesChanged(string propertyName = null)
        {
            if (PropertyChanged != null)
                PropertyChanged(this, new PropertyChangedEventArgs(propertyName));
        }
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public abstract class DashboardProperty
    {
        public const string SerializationSeparator = "#";

        private string m_propertyId;
        public PropertyInfo PropertyInfo { get; set; }

        public abstract ProxyProperty Proxy { get; }

        [YAXSerializableField]
        public string PropertyId
        {
            get
            {
                if (m_propertyId == null)
                    InitPropertyId();
                return m_propertyId;
            }
            internal set { m_propertyId = value; }
        }

        protected abstract void InitPropertyId();

        public abstract void RestoreFromId(MyProject project);

        public abstract bool Equals(object owner, string propertyName);
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class DashboardNodeProperty : DashboardProperty
    {
        public MyNode Node { get; set; }

        public sealed override ProxyProperty Proxy
        {
            get
            {
                var proxy = GetProxyBase();
                proxy.Description = GetDescription();
                proxy.Category = Node.Name;
                proxy.ReadOnly = IsReadonly();

                return proxy;
            }
        }

        protected virtual ProxyProperty GetProxyBase()
        {
            return new ProxyProperty(Node, PropertyInfo)
            {
                Name = GetDisplayName(),
            };
        }

        protected string GetDescription()
        {
            string description = string.Empty;
            var descriptionAttr = PropertyInfo.GetCustomAttribute<DescriptionAttribute>();
            if (descriptionAttr != null)
                description = descriptionAttr.Description;
            return description;
        }

        protected bool IsReadonly()
        {
            var descriptionAttr = PropertyInfo.GetCustomAttribute<ReadOnlyAttribute>();
            if (descriptionAttr != null)
                return descriptionAttr.IsReadOnly;

            return false;
        }

        protected string GetDisplayName()
        {
            string displayName = PropertyInfo.Name;

            var displayAttr = PropertyInfo.GetCustomAttribute<DisplayNameAttribute>();
            if (displayAttr != null)
                displayName = displayAttr.DisplayName;
            return displayName;
        }

        protected override void InitPropertyId()
        {
            PropertyId = string.Join(SerializationSeparator, Node.Id, PropertyInfo.Name);
        }

        public override void RestoreFromId(MyProject project)
        {
            string[] idSplit = PropertyId.Split(SerializationSeparator.ToCharArray());

            var success = false;

            int nodeId;
            if (int.TryParse(idSplit[0], out nodeId))
            {
                Node = project.GetNodeById(nodeId);
                if (Node != null)
                    success = true;
            }

            if (!success)
                throw new SerializationException("A dashboard property target node was not found.");

            PropertyInfo = Node.GetType().GetProperty(idSplit[1]);

            if (PropertyInfo == null)
                throw new SerializationException("A dashboard property was not found on the node.");
        }

        public override bool Equals(object owner, string propertyName)
        {
            return owner == Node && PropertyInfo.Name == propertyName;
        }
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class DashboardTaskProperty : DashboardNodeProperty
    {
        public MyTask Task { get; set; }

        // Task properties are grouped together with node properties.
        protected override ProxyProperty GetProxyBase()
        {
            return new ProxyProperty(Task, PropertyInfo)
            {
                Name = Task.Name + "." + GetDisplayName(),
            };
        }

        protected override void InitPropertyId()
        {
            PropertyId = string.Join(SerializationSeparator, Node.Id, Task.PropertyName, PropertyInfo.Name);
        }

        public override void RestoreFromId(MyProject project)
        {
            string[] idSplit = PropertyId.Split(SerializationSeparator.ToCharArray());

            var success = false;

            int nodeId;
            if (int.TryParse(idSplit[0], out nodeId))
            {
                Node = project.GetNodeById(nodeId);
                if (Node != null)
                    success = true;
            }

            if (!success)
                throw new SerializationException("A task dashboard property did not find the specified node.");

            var workingNode = Node as MyWorkingNode;
            if (workingNode == null)
                throw new SerializationException("A task dashboard property found a node without tasks.");

            Task = workingNode.GetTaskByPropertyName(idSplit[1]);

            if (Task == null)
                throw new SerializationException("A task dashboard property did not find the target task.");

            PropertyInfo = Task.GetType().GetProperty(idSplit[2]);

            if (PropertyInfo == null)
                throw new SerializationException("A task dashboard property was not found on the task.");
        }

        public override bool Equals(object owner, string propertyName)
        {
            return owner == Task && PropertyInfo.Name == propertyName;
        }
    }
}
