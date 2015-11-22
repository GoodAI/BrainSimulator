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
        public TProxy Proxy { get; private set; }
        public ProxyPropertyDescriptorBase(ref TProxy proxy, Attribute[] attrs)
            : base(proxy.Name, attrs)
        {
            Proxy = proxy;
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
            return Proxy.Value;
        }

        public override string Description
        {
            get { return Proxy.Description; }
        }

        public override string Category
        {
            get { return Proxy.Category; }
        }

        public override string DisplayName
        {
            get { return Proxy.Name; }
        }

        public override bool IsReadOnly
        {
            get { return Proxy.ReadOnly; }
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
            Proxy.Value = value;
        }

        public override Type PropertyType
        {
            get
            {
                return Proxy.Value == null ? typeof (string) : Proxy.Value.GetType();
            }
        }

        #endregion
    }

    public sealed class ProxyPropertyDescriptor :
        ProxyPropertyDescriptorBase<SingleProxyProperty, DashboardNodeProperty>
    {
        public ProxyPropertyDescriptor(ref SingleProxyProperty proxy, Attribute[] attrs) : base(ref proxy, attrs)
        {
        }
    }

    public sealed class ProxyPropertyGroupDescriptor :
        ProxyPropertyDescriptorBase<ProxyPropertyGroup, DashboardPropertyGroup>
    {
        public ProxyPropertyGroupDescriptor(ref ProxyPropertyGroup proxy, Attribute[] attrs) : base(ref proxy, attrs)
        {
        }
    }
}
