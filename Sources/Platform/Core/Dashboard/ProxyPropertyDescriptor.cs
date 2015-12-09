using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Core.Dashboard
{
    public class ProxyPropertyDescriptor : PropertyDescriptor
    {
        private TypeConverter m_converter;
        public ProxyPropertyBase Proxy { get; private set; }
        public ProxyPropertyDescriptor(ref ProxyPropertyBase proxy, Attribute[] attrs)
            : base(proxy.Name, attrs)
        {
            Proxy = proxy;
        }

        #region PropertyDescriptor specific

        public void SetConverter(TypeConverter converter)
        {
            m_converter = converter;
        }

        public override bool CanResetValue(object component)
        {
            return false;
        }

        public override Type ComponentType
        {
            get { return null; }
        }

        public override TypeConverter Converter
        {
            get { return m_converter ?? base.Converter; }
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
                return Proxy.Value == null ? typeof(string) : Proxy.Type;
            }
        }

        #endregion
    }
}
