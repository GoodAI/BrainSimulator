using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Core.Dashboard
{
    public abstract class ProxyPropertyDescriptorBase<TProxy, TProperty> : PropertyDescriptor
        where TProxy : ProxyPropertyBase<TProperty>
        where TProperty : DashboardProperty
    {
        public TProxy Property { get; private set; }
        public ProxyPropertyDescriptorBase(ref TProxy property, Attribute[] attrs)
            : base(property.Name, attrs)
        {
            Property = property;
        }

        #region PropertyDescriptor specific


        public override bool CanResetValue(object component)
        {
            return false;
        }

        public override Type ComponentType
        {
            get { return null; }
        }

        public override object GetValue(object component)
        {
            return Property.Value;
        }

        public override string Description
        {
            get { return Property.Description; }
        }

        public override string Category
        {
            get { return Property.Category; }
        }

        public override string DisplayName
        {
            get { return Property.Name; }
        }

        public override bool IsReadOnly
        {
            get { return Property.ReadOnly; }
        }

        public override void ResetValue(object component)
        {
            // TODO(HonzaS): Implement if needed.
        }

        public override bool ShouldSerializeValue(object component)
        {
            return false;
        }

        public override void SetValue(object component, object value)
        {
            Property.Value = value;
        }

        public override Type PropertyType
        {
            get
            {
                return Property.Value == null ? typeof (string) : Property.Value.GetType();
            }
        }

        #endregion
    }

    public sealed class ProxyPropertyDescriptor :
        ProxyPropertyDescriptorBase<SingleProxyProperty, DashboardNodeProperty>
    {
        public ProxyPropertyDescriptor(ref SingleProxyProperty property, Attribute[] attrs) : base(ref property, attrs)
        {
        }
    }

    public sealed class ProxyPropertyGroupDescriptor :
        ProxyPropertyDescriptorBase<ProxyPropertyGroup, DashboardPropertyGroup>
    {
        public ProxyPropertyGroupDescriptor(ref ProxyPropertyGroup property, Attribute[] attrs) : base(ref property, attrs)
        {
        }
    }
}
