using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using YAXLib;

namespace GoodAI.Platform.Core.Dashboard
{
    public class DashboardViewModel : ICustomTypeDescriptor
    {
        private readonly Dashboard m_dashboard;

        public DashboardViewModel(Dashboard dashboard)
        {
            m_dashboard = dashboard;
        }

        #region "TypeDescriptor Implementation"
        /// <summary>
        /// Get Class Name
        /// </summary>
        /// <returns>String</returns>
        public String GetClassName()
        {
            return TypeDescriptor.GetClassName(this, true);
        }

        /// <summary>
        /// GetAttributes
        /// </summary>
        /// <returns>AttributeCollection</returns>
        public AttributeCollection GetAttributes()
        {
            return TypeDescriptor.GetAttributes(this, true);
        }

        /// <summary>
        /// GetComponentName
        /// </summary>
        /// <returns>String</returns>
        public String GetComponentName()
        {
            return TypeDescriptor.GetComponentName(this, true);
        }

        /// <summary>
        /// GetConverter
        /// </summary>
        /// <returns>TypeConverter</returns>
        public TypeConverter GetConverter()
        {
            return TypeDescriptor.GetConverter(this, true);
        }

        /// <summary>
        /// GetDefaultEvent
        /// </summary>
        /// <returns>EventDescriptor</returns>
        public EventDescriptor GetDefaultEvent()
        {
            return TypeDescriptor.GetDefaultEvent(this, true);
        }

        /// <summary>
        /// GetDefaultProperty
        /// </summary>
        /// <returns>PropertyDescriptor</returns>
        public PropertyDescriptor GetDefaultProperty()
        {
            return TypeDescriptor.GetDefaultProperty(this, true);
        }

        /// <summary>
        /// GetEditor
        /// </summary>
        /// <param name="editorBaseType">editorBaseType</param>
        /// <returns>object</returns>
        public object GetEditor(Type editorBaseType)
        {
            return TypeDescriptor.GetEditor(this, editorBaseType, true);
        }

        public EventDescriptorCollection GetEvents(Attribute[] attributes)
        {
            return TypeDescriptor.GetEvents(this, attributes, true);
        }

        public EventDescriptorCollection GetEvents()
        {
            return TypeDescriptor.GetEvents(this, true);
        }

        public PropertyDescriptorCollection GetProperties(Attribute[] attributes)
        {
            return new PropertyDescriptorCollection(
                m_dashboard.Properties.Select(property =>
                {
                    var proxy = property.Proxy;
                    return new ProxyPropertyDescriptor(ref proxy, attributes);
                }).Cast<PropertyDescriptor>().ToArray());
        }

        public PropertyDescriptorCollection GetProperties()
        {

            return TypeDescriptor.GetProperties(this, true);

        }

        public object GetPropertyOwner(PropertyDescriptor pd)
        {
            return this;
        }
        #endregion
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class Dashboard
    {
        [YAXSerializableField]
        public IList<DashboardProperty> Properties { get; set; }

        public Dashboard()
        {
            Properties = new List<DashboardProperty>();
        }

        public void RestoreFromIds(MyProject project)
        {
            foreach (var property in Properties)
                property.RestoreFromId(project);
        }
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public abstract class DashboardProperty
    {
        public const string Separator = "#";

        private string m_propertyId;
        public PropertyInfo Property { get; set; }

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
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class DashboardNodeProperty : DashboardProperty
    {
        public MyNode Node { get; set; }

        public override ProxyProperty Proxy
        {
            get {
                return new ProxyProperty(Node, Property)
                {
                    Name = Property.Name,
                    Category = Node.Name
                };
            }
        }

        protected override void InitPropertyId()
        {
            PropertyId = string.Join(Separator, Node.Id, Property.Name);
        }

        public override void RestoreFromId(MyProject project)
        {
            string[] idSplit = PropertyId.Split(Separator.ToCharArray());

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

            Property = Node.GetType().GetProperty(idSplit[1]);

            if (Property == null)
                throw new SerializationException("A dashboard property was not found on the node.");
        }
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class DashboardTaskProperty : DashboardNodeProperty
    {
        public MyTask Task { get; set; }

        // Task properties are grouped together with node properties.
        public override ProxyProperty Proxy
        {
            get
            {
                return new ProxyProperty(Task, Property)
                {
                    Name = Task.Name + Separator + Property.Name,
                    Category = Task.GenericOwner.Name
                };
            }
        }

        protected override void InitPropertyId()
        {
            PropertyId = string.Join(Separator, Node.Id, Task.Name, Property.Name);
        }

        public override void RestoreFromId(MyProject project)
        {
            string[] idSplit = PropertyId.Split(Separator.ToCharArray());

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

            Property = Task.GetType().GetProperty(idSplit[2]);

            if (Property == null)
                throw new SerializationException("A task dashboard property was not found on the task.");
        }
    }
}
