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
    public class DashboardViewModelBase<TDashboard, TProperty> : ICustomTypeDescriptor, INotifyPropertyChanged
        where TDashboard : DashboardBase<TProperty>
        where TProperty : DashboardProperty
    {
        protected readonly TDashboard Dashboard;

        public event PropertyChangedEventHandler PropertyChanged
        {
            add { Dashboard.PropertyChanged += value; }
            remove { Dashboard.PropertyChanged -= value; }
        }

        public DashboardViewModelBase(TDashboard dashboard)
        {
            Dashboard = dashboard;
        }

        public void RemoveProperty(ProxyPropertyBase proxy)
        {
            var proxyWithSource = proxy as ProxyPropertyBase<TProperty>;
            Dashboard.Remove(proxyWithSource.SourceProperty);
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
                Dashboard.Properties
                    .Select(property => property.Proxy)
                    .Where(proxy => proxy.Visible)
                    .Select(proxy => new ProxyPropertyDescriptor(ref proxy, attributes))
                    .Cast<PropertyDescriptor>()
                    .ToArray());
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

    public class DashboardViewModel : DashboardViewModelBase<Dashboard, DashboardNodeProperty>
    {
        public DashboardViewModel(Dashboard dashboard) : base(dashboard)
        {
        }
    }

    public class GroupedDashboardViewModel : DashboardViewModelBase<GroupDashboard, DashboardPropertyGroup>
    {
        public GroupedDashboardViewModel(GroupDashboard dashboard) : base(dashboard)
        {
        }

        public void AddGroupedProperty()
        {
            Dashboard.AddGroupedProperty();
        }
    }
}
