using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Core.Dashboard
{
    public abstract class ProxyPropertyBase
    {
        public string Name { get; set; }
        public string Description { get; set; }
        public bool ReadOnly { get; set; }
        public bool Visible { get; set; }

        public string Category { get; set; }
        public abstract object Value { get; set; }

        public DashboardProperty SourceProperty { get; private set; }

        public ProxyPropertyBase(DashboardProperty sourceProperty)
        {
            SourceProperty = sourceProperty;
        }
    }

    public sealed class SingleProxyProperty : ProxyPropertyBase
    {
        public PropertyInfo PropertyInfo { get; private set; }
        public object Target { get; protected set; }

        public SingleProxyProperty(DashboardProperty sourceProperty, object target, PropertyInfo propertyInfo)
            : base(sourceProperty)
        {
            Target = target;
            PropertyInfo = propertyInfo;
            Visible = true;
        }

        public override object Value
        {
            get { return PropertyInfo.GetValue(Target); }
            set { PropertyInfo.SetValue(Target, value); }
        }
    }

    public sealed class GroupedProxyProperty : ProxyPropertyBase
    {
        private object m_value;

        public GroupedProxyProperty(DashboardGroupedProperty sourceProperty) : base(sourceProperty)
        {
        }

        public override object Value
        {
            get { return m_value; }
            set
            {
                m_value = value;
                foreach (var property in (SourceProperty as DashboardGroupedProperty).GroupedProperties)
                {
                    property.Proxy.Value = value;
                }
            }
        }
    }
}
