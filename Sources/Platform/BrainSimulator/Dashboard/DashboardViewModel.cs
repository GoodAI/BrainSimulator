using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Dashboard;

namespace GoodAI.BrainSimulator.DashboardUtils
{
    public class DashboardViewModel : ICustomTypeDescriptor, INotifyPropertyChanged
    {
        private readonly Dashboard m_dashboard;

        public event PropertyChangedEventHandler PropertyChanged
        {
            add { m_dashboard.PropertyChanged += value; }
            remove { m_dashboard.PropertyChanged -= value; }
        }

        public DashboardViewModel(Dashboard dashboard)
        {
            m_dashboard = dashboard;
        }

        public void RemoveProperty(object owner, string propertyName)
        {
            m_dashboard.Remove(owner, propertyName);
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
}
