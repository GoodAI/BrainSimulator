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

        public abstract ProxyPropertyBase Proxy { get; }
        public abstract object Target { get; }
        public abstract string PropertyName { get; }

        /// <summary>
        /// This is only used for serialization, do not access it otherwise.
        /// </summary>
        [YAXSerializableField]
        protected internal string PropertyId
        {
            get
            {
                if (m_propertyId == null)
                    m_propertyId = GeneratePropertyId();
                return m_propertyId;
            }
            set { m_propertyId = value; }
        }

        protected abstract string GeneratePropertyId();

        public abstract void Restore(MyProject project);
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class DashboardNodeProperty : DashboardProperty
    {
        public PropertyInfo PropertyInfo { get; set; }
        public MyNode Node { get; set; }

        private DashboardPropertyGroup m_group;
        public DashboardPropertyGroup Group
        {
            get { return m_group; }
            set
            {
                m_group = value;
                GroupId = value.Name;
            }
        }

        [YAXSerializableField]
        public string GroupId { get; internal set; }

        private SingleProxyProperty m_proxy;
        public sealed override ProxyPropertyBase Proxy
        {
            get
            {
                if (m_proxy == null)
                {
                    m_proxy = GetProxyBase();
                    m_proxy.Description = GetDescription();
                    m_proxy.Category = Node.Name;
                }

                return m_proxy;
            }
        }

        public override string PropertyName
        {
            get { return PropertyInfo.Name; }
        }

        public override object Target
        {
            get { return Node; }
        }

        protected virtual SingleProxyProperty GetProxyBase()
        {
            return new SingleProxyProperty(this, Node, PropertyInfo)
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

        public bool IsReadonly()
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

        protected override string GeneratePropertyId()
        {
            return string.Join(SerializationIdSeparator, Node.Id, PropertyInfo.Name);
        }

        public override void Restore(MyProject project)
        {
            string[] idSplit = PropertyId.Split(SerializationIdSeparator.ToCharArray());

            var success = false;

            int nodeId;
            if (int.TryParse(idSplit[0], out nodeId))
            {
                Node = project.GetNodeById(nodeId);
                if (Node != null)
                    success = true;
            }

            if (!success)
                throw new SerializationException("A dashboard property target node was not found");

            PropertyInfo = Node.GetType().GetProperty(idSplit[1]);

            if (PropertyInfo == null)
                throw new SerializationException("A dashboard property was not found on the node");
        }

        public virtual bool Equals(object owner, string propertyName)
        {
            return owner == Node && PropertyInfo.Name == propertyName;
        }
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class DashboardTaskProperty : DashboardNodeProperty
    {
        public MyTask Task { get; set; }

        // Task properties are grouped together with node properties.
        protected override SingleProxyProperty GetProxyBase()
        {
            return new SingleProxyProperty(this, Task, PropertyInfo)
            {
                Name = Task.Name + "." + GetDisplayName(),
            };
        }

        protected override string GeneratePropertyId()
        {
            return string.Join(SerializationIdSeparator, Node.Id, Task.PropertyName, PropertyInfo.Name);
        }

        public override object Target
        {
            get { return Task; }
        }

        public override void Restore(MyProject project)
        {
            base.Restore(project);

            string[] idSplit = PropertyId.Split(SerializationIdSeparator.ToCharArray());

            var success = false;

            int nodeId;
            if (int.TryParse(idSplit[0], out nodeId))
            {
                Node = project.GetNodeById(nodeId);
                if (Node != null)
                    success = true;
            }

            if (!success)
                throw new SerializationException("A task dashboard property did not find the specified node");

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

        public override bool Equals(object owner, string propertyName)
        {
            return owner == Task && PropertyInfo.Name == propertyName;
        }
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public sealed class DashboardPropertyGroup : DashboardProperty
    {
        private IList<string> m_groupedPropertyIds;
        public string Name { get; set; }

        public IList<DashboardNodeProperty> GroupedProperties { get; private set; }

        ProxyPropertyBase m_proxy;
        public override ProxyPropertyBase Proxy
        {
            get
            {
                if (m_proxy == null)
                {
                    m_proxy = new ProxyPropertyGroup(this)
                    {
                        Name = Name
                    };
                }

                return m_proxy;
            }
        }

        public override object Target
        {
            get { return this; }
        }

        public override string PropertyName
        {
            get { return Name; }
        }

        protected override string GeneratePropertyId()
        {
            return Name;
        }

        public DashboardPropertyGroup()
        {
            GroupedProperties = new List<DashboardNodeProperty>();
        }

        public override void Restore(MyProject project)
        {
            Name = PropertyId;

            foreach (var property in project.Dashboard.Properties)
            {
                if (property.GroupId == PropertyId)
                    Add(property);
            }
        }

        public void CheckType(DashboardNodeProperty property)
        {
            if (GroupedProperties.Any() &&
                property.PropertyInfo.PropertyType != GroupedProperties.First().PropertyInfo.PropertyType)
                throw new InvalidOperationException(string.Format("Wrong property type: {0}",
                    property.PropertyInfo.PropertyType));
        }

        public void Add(DashboardNodeProperty property)
        {
            if (GroupedProperties.Contains(property))
                return;

            GroupedProperties.Add(property);
            property.Group = this;
            property.Proxy.Value = GroupedProperties.First().Proxy.Value;
        }

        public void Remove(DashboardNodeProperty property)
        {
            GroupedProperties.Remove(property);
            property.Group = null;
        }

        public bool Contains(DashboardNodeProperty property)
        {
            return GroupedProperties.Contains(property);
        }
    }
}
