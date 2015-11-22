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
        public abstract string Name { get; }

        public virtual string FullName { get { return Name; } }

        public virtual string Description { get; set; }
        public bool ReadOnly { get; set; }
        public virtual bool IsVisible { get; set; }

        public string Category { get; set; }
        public abstract object Value { get; set; }

        public ProxyPropertyBase()
        {
            IsVisible = true;
        }

        public override string ToString()
        {
            return FullName.Replace("\t", "");
        }
    }

    public abstract class ProxyPropertyBase<TSource> : ProxyPropertyBase where TSource : DashboardProperty
    {
        public TSource SourceProperty { get; private set; }

        public ProxyPropertyBase(TSource sourceProperty)
        {
            SourceProperty = sourceProperty;
        }

        public override string Name {
            get { return SourceProperty.DisplayName; }
        }
    }

    public sealed class SingleProxyProperty : ProxyPropertyBase<DashboardNodeProperty>
    {
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

        public override string FullName { get { return SourceProperty.Node.Name + "." + Name; } }
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

        public IEnumerable<SingleProxyProperty> GetGroupMembers()
        {
            return SourceProperty.GroupedProperties.Select(member => member.Proxy);
        }
    }
}
