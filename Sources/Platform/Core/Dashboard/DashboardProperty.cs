using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Dashboard;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using YAXLib;

namespace GoodAI.Core.Dashboard
{
    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public abstract class DashboardProperty
    {
        public const string SerializationIdSeparator = "#";

        private string m_propertyId;
        public abstract ProxyPropertyBase GenericProxy { get; }
        public abstract string PropertyName { get; set; }
        public abstract string DisplayName { get; }

        /// <summary>
        /// This is only used for serialization, do not access it otherwise.
        /// </summary>
        [YAXSerializableField]
        public string PropertyId
        {
            get
            {
                if (m_propertyId == null)
                    m_propertyId = GeneratePropertyId();
                return m_propertyId;
            }
            protected set { m_propertyId = value; }
        }

        protected abstract string GeneratePropertyId();

        public abstract void Restore(MyProject project);

        public abstract override bool Equals(object obj);
    }

    public abstract class DashboardNodePropertyBase : DashboardProperty
    {
        public MyNode Node { get; set; }

        public abstract object Target { get; }
        private DashboardPropertyGroup m_group;
        public DashboardPropertyGroup Group
        {
            get { return m_group; }
            set
            {
                m_group = value;
                GroupId = value == null ? null : value.PropertyId;
            }
        }

        [YAXSerializableField]
        public string GroupId { get; internal set; }

        private ProxyPropertyBase m_proxy;

        public sealed override ProxyPropertyBase GenericProxy
        {
            get
            {
                if (m_proxy == null)
                    m_proxy = GetProxyBase();

                return m_proxy;
            }
        }

        protected static MyNode FindNode(MyProject project, string nodeId)
        {
            var success = false;

            MyNode node = null;
            int parsedNodeId;
            if (int.TryParse(nodeId, out parsedNodeId))
            {
                node = project.GetNodeById(parsedNodeId);
                if (node != null)
                    success = true;
            }

            if (!success)
                throw new SerializationException("A dashboard property target node was not found");

            return node;
        }

        protected abstract ProxyPropertyBase GetProxyBase();
        public virtual bool IsReadOnly { get; set; }
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class DashboardNodeProperty : DashboardNodePropertyBase
    {
        private bool? m_readonly;
        private string m_displayName;

        public override object Target { get { return Node; } }

        public PropertyInfo PropertyInfo { get; protected set; }

        public DashboardNodeProperty(MyNode node, PropertyInfo propertyInfo)
        {
            Node = node;
            PropertyInfo = propertyInfo;
        }

        public DashboardNodeProperty() { }

        protected override ProxyPropertyBase GetProxyBase()
        {
            return new SingleProxyProperty(this, Node, PropertyInfo);
        }

        protected override string GeneratePropertyId()
        {
            return string.Join(SerializationIdSeparator, Node.Id, PropertyInfo.Name);
        }

        public override bool IsReadOnly
        {
            get
            {
                if (m_readonly == null)
                {
                    var descriptionAttr = PropertyInfo.GetCustomAttribute<ReadOnlyAttribute>();
                    m_readonly = descriptionAttr != null && descriptionAttr.IsReadOnly;
                }

                return m_readonly.Value;
            }
        }

        public override void Restore(MyProject project)
        {
            string[] idSplit = PropertyId.Split(SerializationIdSeparator.ToCharArray());

            Node = FindNode(project, idSplit[0]);

            PropertyInfo = Node.GetType().GetProperty(idSplit[1]);

            if (PropertyInfo == null)
                throw new SerializationException("A dashboard property was not found on the node");
        }

        public override bool Equals(object obj)
        {
            var o = obj as DashboardNodeProperty;
            if (o != null)
                return Node == o.Node && PropertyName == o.PropertyName;

            return false;
        }

        public override string PropertyName
        {
            get { return PropertyInfo.Name; }
            set { throw new InvalidOperationException("Cannot set name of a delegate property."); }
        }

        public override string DisplayName
        {
            get
            {
                if (m_displayName == null)
                {
                    m_displayName = PropertyInfo.Name;

                    var displayAttr = PropertyInfo.GetCustomAttribute<DisplayNameAttribute>();
                    if (displayAttr != null)
                        m_displayName = displayAttr.DisplayName;
                }

                return m_displayName;
            }
        }
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class DashboardTaskGroupProperty : DashboardNodePropertyBase
    {
        public string GroupName { get { return TaskGroup.GroupName; } }

        public TaskGroup TaskGroup { get; private set; }

        public override object Target { get { return TaskGroup; } }

        private MyWorkingNode WorkingNode
        {
            get { return Node as MyWorkingNode; }
            set { Node = value; }
        }

        public DashboardTaskGroupProperty(TaskGroup taskGroup)
        {
            TaskGroup = taskGroup;
            WorkingNode = TaskGroup.Owner;
        }

        public DashboardTaskGroupProperty() { }

        protected override ProxyPropertyBase GetProxyBase()
        {
            return new TaskGroupProxyProperty(this, GroupName);
        }

        public override string PropertyName
        {
            // We don't really need the property name here as the group is uniquely identified by the Target.
            // However, this makes it easier during debugging etc.
            get { return GroupName; }
            set { throw new InvalidOperationException("Cannot set name of a delegate property."); }
        }

        public override string DisplayName
        {
            get { return "[Task Group] " + GroupName; }
        }

        protected override string GeneratePropertyId()
        {
            return string.Join(SerializationIdSeparator, Node.Id, GroupName);
        }

        public override void Restore(MyProject project)
        {
            string[] idSplit = PropertyId.Split(new[] { SerializationIdSeparator }, StringSplitOptions.RemoveEmptyEntries);

            MyNode node = FindNode(project, idSplit[0]);
            WorkingNode = node as MyWorkingNode;
            if (Node == null)
                throw new SerializationException(string.Format("Node id {0} was found but doesn't contain tasks", node.Id));

            string taskGroupName = idSplit[1];
            TaskGroup taskGroup;
            if (!WorkingNode.TaskGroups.TryGetValue(taskGroupName, out taskGroup))
                throw new SerializationException(string.Format("Task group {0} not found", taskGroupName));

            TaskGroup = taskGroup;
        }

        public override bool Equals(object obj)
        {
            var o = obj as DashboardTaskGroupProperty;
            if (o != null)
                return Node == o.Node && GroupName == o.GroupName;

            return false;
        }
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class DashboardTaskProperty : DashboardNodeProperty
    {
        public MyTask Task { get; set; }

        public override object Target { get { return Task; } }

        public DashboardTaskProperty(MyTask task, PropertyInfo propertyInfo) : base(task.GenericOwner, propertyInfo)
        {
            Task = task;
        }

        public DashboardTaskProperty() { }

        // Task properties are grouped together with node properties.
        protected override ProxyPropertyBase GetProxyBase()
        {
            return new SingleProxyProperty(this, Task, PropertyInfo);
        }

        public override string DisplayName
        {
            get { return Task.Name + "." + base.DisplayName; }
        }

        protected override string GeneratePropertyId()
        {
            return string.Join(SerializationIdSeparator, Node.Id, Task.PropertyName, PropertyInfo.Name);
        }

        public override void Restore(MyProject project)
        {
            string[] idSplit = PropertyId.Split(new[] { SerializationIdSeparator }, StringSplitOptions.RemoveEmptyEntries);

            Node = FindNode(project, idSplit[0]);

            var workingNode = Node as MyWorkingNode;
            if (workingNode == null)
                throw new SerializationException("A task dashboard property found a node without tasks");

            Task = workingNode.GetTaskByPropertyName(idSplit[1]);

            if (Task == null)
                throw new SerializationException("A task dashboard property did not find the target task");

            PropertyInfo = Task.GetType().GetProperty(idSplit[2]);

            if (PropertyInfo == null)
                throw new SerializationException("A task dashboard property was not found on the task");
        }

        public override bool Equals(object obj)
        {
            var o = obj as DashboardTaskProperty;
            if (o != null)
                return Task == o.Task && PropertyName == o.PropertyName;

            return false;
        }
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public sealed class DashboardPropertyGroup : DashboardProperty
    {
        public IList<DashboardNodePropertyBase> GroupedProperties { get; private set; }

        private ProxyPropertyBase m_proxy;

        public override ProxyPropertyBase GenericProxy
        {
            get
            {
                if (m_proxy == null)
                    m_proxy = new ProxyPropertyGroup(this);

                return m_proxy;
            }
        }

        [YAXSerializableField]
        public override string PropertyName { get; set; }

        public override string DisplayName
        {
            get
            {
                string type = null;
                var firstProperty = GroupedProperties.FirstOrDefault();
                if (firstProperty != null)
                    type = firstProperty.GenericProxy.TypeName + ", ";

                return PropertyName + string.Format(" ({0}{1})", type, GroupedProperties.Count);
            }
        }

        protected override string GeneratePropertyId()
        {
            return Guid.NewGuid().ToString();
        }

        public DashboardPropertyGroup(string name) : this()
        {
            PropertyName = name;
            PropertyId = Guid.NewGuid().ToString();
        }

        public DashboardPropertyGroup()
        {
            GroupedProperties = new List<DashboardNodePropertyBase>();
        }

        public override void Restore(MyProject project)
        {
            foreach (DashboardNodePropertyBase property in project.Dashboard.Properties.Where(p => p.GroupId == PropertyId))
                Add(property);
        }

        private void CheckType(DashboardNodePropertyBase property)
        {
            if (GroupedProperties.Any() && !GroupedProperties.First().GenericProxy.CompatibleWith(property.GenericProxy))
                throw new InvalidOperationException(string.Format("Wrong property type: {0}", property.GenericProxy.Type));
        }

        public void Add(DashboardNodePropertyBase property)
        {
            CheckType(property);

            if (GroupedProperties.Contains(property))
                return;

            GroupedProperties.Add(property);
            property.Group = this;
            property.GenericProxy.Value = GroupedProperties.First().GenericProxy.Value;
        }

        public void Remove(DashboardNodePropertyBase property)
        {
            GroupedProperties.Remove(property);
            property.Group = null;
        }

        public void Clear()
        {
            foreach (var property in GroupedProperties)
                property.Group = null;

            GroupedProperties.Clear();
        }

        public override bool Equals(object obj)
        {
            return this == obj;
        }
    }
}
