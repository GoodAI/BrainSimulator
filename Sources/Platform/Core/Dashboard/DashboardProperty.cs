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
        public abstract object Target { get; }
        public abstract string PropertyName { get; set; }
        public abstract string DisplayName { get; }

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
                GroupId = value == null ? null : value.PropertyId;
            }
        }

        [YAXSerializableField]
        public string GroupId { get; internal set; }

        private SingleProxyProperty m_proxy;
        private string m_displayName;
        private bool? m_readonly;

        public sealed override ProxyPropertyBase GenericProxy
        {
            get
            {
                if (m_proxy == null)
                {
                    string description = string.Empty;
                    var descriptionAttr = PropertyInfo.GetCustomAttribute<DescriptionAttribute>();
                    if (descriptionAttr != null)
                        description = descriptionAttr.Description;

                    m_proxy = GetProxyBase();
                    m_proxy.Description = description;
                }

                return m_proxy;
            }
        }

        public SingleProxyProperty Proxy { get { return GenericProxy as SingleProxyProperty; } }

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

        public override object Target
        {
            get { return Node; }
        }

        protected virtual SingleProxyProperty GetProxyBase()
        {
            return new SingleProxyProperty(this, Node, PropertyInfo);
        }

        public bool IsReadonly
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
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class DashboardTaskProperty : DashboardNodeProperty
    {
        public MyTask Task { get; set; }

        // Task properties are grouped together with node properties.
        protected override SingleProxyProperty GetProxyBase()
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

        public override object Target
        {
            get { return Task; }
        }

        public override void Restore(MyProject project)
        {
            string[] idSplit = PropertyId.Split(new [] {SerializationIdSeparator}, StringSplitOptions.RemoveEmptyEntries);

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
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public sealed class DashboardPropertyGroup : DashboardProperty
    {
        public IList<DashboardNodeProperty> GroupedProperties { get; private set; }

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

        public ProxyPropertyGroup Proxy { get { return GenericProxy as ProxyPropertyGroup; } }

        public override object Target
        {
            get { return this; }
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
                    type = firstProperty.Proxy.Value.GetType().Name + ", ";

                return PropertyName + string.Format(" ({0}{1})", type, GroupedProperties.Count);
            }
        }

        protected override string GeneratePropertyId()
        {
            return Guid.NewGuid().ToString();
        }

        public DashboardPropertyGroup()
        {
            GroupedProperties = new List<DashboardNodeProperty>();
            PropertyId = Guid.NewGuid().ToString();
        }

        public override void Restore(MyProject project)
        {
            foreach (DashboardNodeProperty property in project.Dashboard.Properties.Where(p => p.GroupId == PropertyId))
                Add(property);
        }

        private void CheckType(DashboardNodeProperty property)
        {
            if (GroupedProperties.Any() &&
                property.PropertyInfo.PropertyType != GroupedProperties.First().PropertyInfo.PropertyType)
                throw new InvalidOperationException(string.Format("Wrong property type: {0}",
                    property.PropertyInfo.PropertyType));
        }

        public void Add(DashboardNodeProperty property)
        {
            CheckType(property);

            if (GroupedProperties.Contains(property))
                return;

            GroupedProperties.Add(property);
            property.Group = this;
            property.GenericProxy.Value = GroupedProperties.First().GenericProxy.Value;
        }

        public void Remove(DashboardNodeProperty property)
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
    }
}
