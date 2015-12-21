using System;
using System.ComponentModel;
using System.Linq;
using GoodAI.Core.Dashboard;

namespace GoodAI.BrainSimulator.DashboardUtils
{
    public abstract class DashboardViewModelBase<TDashboard, TProperty> : ICustomTypeDescriptor, INotifyPropertyChanged
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

        public virtual void RemoveProperty(ProxyPropertyBase proxy)
        {
            Dashboard.Remove(proxy.GenericSourceProperty as TProperty);
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
                    .Where(property => property.GenericProxy.IsVisible)
                    .Select(property => GetDescriptor(property, attributes))
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

        protected abstract PropertyDescriptor GetDescriptor(TProperty property, Attribute[] attributes);

        public void RemovePropertyOf(object owner)
        {
            Dashboard.RemoveAll(owner);
        }

        public TProperty GetProperty(string propertyId)
        {
            return Dashboard.Get(propertyId);
        }
    }

    public class DashboardViewModel : DashboardViewModelBase<Dashboard, DashboardNodePropertyBase>
    {
        public DashboardViewModel(Dashboard dashboard) : base(dashboard)
        {
        }

        protected override PropertyDescriptor GetDescriptor(DashboardNodePropertyBase property, Attribute[] attributes)
        {
            ProxyPropertyBase proxy = property.GenericProxy;
            return new ProxyPropertyDescriptor(ref proxy, attributes);
        }

        public DashboardNodePropertyBase GetProperty(object target, string propertyName)
        {
            return Dashboard.Get(target, propertyName);
        }
    }

    public class GroupedDashboardViewModel : DashboardViewModelBase<GroupDashboard, DashboardPropertyGroup>
    {
        public GroupedDashboardViewModel(GroupDashboard dashboard) : base(dashboard)
        {
        }

        public void AddGroupedProperty()
        {
            Dashboard.Add();
        }

        protected override PropertyDescriptor GetDescriptor(DashboardPropertyGroup property, Attribute[] attributes)
        {
            ProxyPropertyBase proxy = property.GenericProxy;
            return new ProxyPropertyDescriptor(ref proxy, attributes);
        }

        public override void RemoveProperty(ProxyPropertyBase proxy)
        {
            Dashboard.Get(proxy.PropertyId).Clear();
            base.RemoveProperty(proxy);
        }
    }
}
