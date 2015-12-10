using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Nodes;

namespace GoodAI.Core.Dashboard
{
    public abstract class ProxyPropertyBase
    {
        private bool m_isVisible;

        public virtual string Name {
            get { return GenericSourceProperty.DisplayName; }
        }

        public DashboardProperty GenericSourceProperty { get; set; }

        public string PropertyId { get { return GenericSourceProperty.PropertyId; } }

        public virtual string FullName { get { return Name; } }

        public virtual string Description { get; set; }
        public bool ReadOnly { get; set; }

        public virtual bool IsVisible
        {
            get { return m_isVisible; }
            set { m_isVisible = value; }
        }

        public virtual string Category { get; set; }
        public abstract object Value { get; set; }

        public abstract Type Type { get; }

        protected ProxyPropertyBase(DashboardProperty property)
        {
            m_isVisible = true;
            GenericSourceProperty = property;
        }

        public override string ToString()
        {
            return FullName.Replace("\t", "");
        }
    }

    public abstract class ProxyPropertyBase<TSource> : ProxyPropertyBase where TSource : DashboardProperty
    {
        protected TSource SourceProperty
        {
            get { return GenericSourceProperty as TSource; }
        }

        public ProxyPropertyBase(DashboardProperty property) : base(property)
        {
        }
    }

    public sealed class SingleProxyProperty : ProxyPropertyBase<DashboardNodeProperty>
    {
        private string m_description;
        public PropertyInfo PropertyInfo { get; private set; }
        public object Target { get; protected set; }

        public override bool IsVisible
        {
            get { return SourceProperty.Group == null; }
        }

        public SingleProxyProperty(DashboardNodeProperty sourceProperty, object target, PropertyInfo propertyInfo)
            : base(sourceProperty)
        {
            Target = target;
            PropertyInfo = propertyInfo;
        }

        public override object Value
        {
            get { return PropertyInfo.GetValue(Target); }
            set { PropertyInfo.SetValue(Target, value); }
        }

        public override Type Type
        {
            get { return PropertyInfo.PropertyType; }
        }

        public override string FullName { get { return SourceProperty.Node.Name + "." + Name; } }

        public override string Category
        {
            get { return SourceProperty.Node.Name; }
        }

        public override string Description
        {
            get
            {
                if (m_description == null)
                {
                    m_description = string.Empty;
                    var descriptionAttr = PropertyInfo.GetCustomAttribute<DescriptionAttribute>();
                    if (descriptionAttr != null)
                        m_description = descriptionAttr.Description;
                }

                return m_description;
            }
        }
    }

    public sealed class TaskGroupProxyProperty : ProxyPropertyBase<DashboardTaskGroupProperty>
    {
        public MyWorkingNode Node { get; set; }
        public string GroupName { get; set; }

        public TaskGroupProxyProperty(DashboardTaskGroupProperty sourceProperty, MyWorkingNode node, string groupName) : base(sourceProperty)
        {
            Node = node;
            GroupName = groupName;
        }

        public override object Value
        {
            get { return Node.GetEnabledTask(GroupName).Name; }
            set { Node.GetTaskByPropertyName(value as string).Enabled = true; }
        }

        public override Type Type
        {
            get { return typeof (string); }
        }
    }

    public sealed class ProxyPropertyGroup : ProxyPropertyBase<DashboardPropertyGroup>
    {
        public ProxyPropertyGroup(DashboardPropertyGroup sourceProperty) : base(sourceProperty)
        {
        }

        public override object Value
        {
            get
            {
                var groupedProperties = SourceProperty.GroupedProperties;
                return groupedProperties.Any() ? groupedProperties.First().GenericProxy.Value : null;
            }
            set
            {
                foreach (var property in SourceProperty.GroupedProperties)
                {
                    property.GenericProxy.Value = value;
                }
            }
        }

        public override string Description
        {
            get { return string.Join(", ", SourceProperty.GroupedProperties.Select(property => property.GenericProxy.FullName)); }
        }

        public override Type Type
        {
            get
            {
                var groupedProperties = SourceProperty.GroupedProperties;
                if (groupedProperties.Any())
                    return groupedProperties.First().GenericProxy.Type;

                return typeof (object);
            }
        }
    }
}
