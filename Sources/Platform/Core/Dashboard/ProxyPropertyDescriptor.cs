using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Task;

namespace GoodAI.Core.Dashboard
{
    public class ProxyPropertyDescriptor : PropertyDescriptor
    {
        public ProxyPropertyBase Proxy { get; private set; }
        public ProxyPropertyDescriptor(ref ProxyPropertyBase proxy, Attribute[] attrs)
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

        public override TypeConverter Converter
        {
            get
            {
                if (Proxy is TaskGroupProxyProperty)
                {
                    var taskNames = (Proxy.GenericSourceProperty as DashboardTaskGroupProperty).TaskGroup.Tasks.Select(task => task.Name);
                    return new TaskGroupConverter(taskNames);
                }

                if (Proxy is ProxyPropertyGroup)
                {
                    // Get the mapping from the first member property.
                    var propertyGroup = Proxy.GenericSourceProperty as DashboardPropertyGroup;

                    if (propertyGroup.GroupedProperties.Any())
                    {
                        var taskGroupProxy = propertyGroup.GroupedProperties.First() as DashboardTaskGroupProperty;
                        if (taskGroupProxy != null)
                        {
                            var taskNames = taskGroupProxy.TaskGroup.Tasks.Select(task => task.Name);
                            return new TaskGroupConverter(taskNames);
                        }
                    }
                }

                return base.Converter;
            }
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
                return Proxy.Type;
            }
        }

        #endregion
    }
}
