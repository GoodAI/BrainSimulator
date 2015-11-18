using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Platform.Core.Dashboard
{
    class ProxyPropertyDescriptor : PropertyDescriptor
    {
        private readonly ProxyProperty m_property;
        public ProxyPropertyDescriptor(ref ProxyProperty property, Attribute[] attrs)
            : base(property.Name, attrs)
        {
            m_property = property;
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
            return m_property.Value;
        }

        public override string Description
        {
            get { return m_property.Name; }
        }

        public override string Category
        {
            get { return m_property.Category; }
        }

        public override string DisplayName
        {
            get { return m_property.Name; }
        }

        public override bool IsReadOnly
        {
            get { return m_property.ReadOnly; }
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
            m_property.Value = value;
        }

        public override Type PropertyType
        {
            get { return m_property.Value.GetType(); }
        }

        #endregion
    }
}
