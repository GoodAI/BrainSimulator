using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using YAXLib;

namespace GoodAI.Core.Dashboard
{
    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class Dashboard : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;

        [YAXSerializableField]
        public IList<DashboardProperty> Properties { get; set; }

        public Dashboard()
        {
            Properties = new List<DashboardProperty>();
        }

        public bool Contains(object target, string propertyName)
        {
            return
                Properties.Select(property => property)
                    .Any(property => property.Target == target && property.PropertyName == propertyName);
        }

        public void RestoreFromIds(MyProject project)
        {
            foreach (var property in Properties.ToList())
            {
                try
                {
                    property.Restore(project);
                }
                catch
                {
                    Properties.Remove(property);
                    MyLog.WARNING.WriteLine("A property with identifier \"{0}\" could not be deserialized.",
                        property.PropertyId);
                }
            }
        }

        public void Add(object target, string propertyName)
        {
            if (Contains(target, propertyName))
                return;

            DashboardNodeProperty property = null;

            var node = target as MyNode;
            if (node != null)
            {
                property = new DashboardNodeProperty
                {
                    Node = node,
                    PropertyInfo = node.GetType().GetProperty(propertyName)
                };
            }
            else
            {
                var task = target as MyTask;
                if (task != null)
                {
                    property = new DashboardTaskProperty
                    {
                        Node = task.GenericOwner,
                        Task = task,
                        PropertyInfo = task.GetType().GetProperty(propertyName)
                    };
                }
            }

            if (property == null)
                throw new InvalidOperationException("Invalid property owner provided");

            if (property.IsReadonly())
                throw new InvalidOperationException("Readonly properties are not supported");

            Properties.Add(property);
            OnPropertiesChanged("Properties");
        }

        public void Remove(DashboardProperty property)
        {
            if (Properties.Remove(property))
                OnPropertiesChanged("Properties");
        }

        public void Remove(object target, string propertyName)
        {
            var property = Properties.FirstOrDefault(p => p.Target == target && p.PropertyName == propertyName);

            if (property == null)
                return;

            if (Properties.Remove(property))
                OnPropertiesChanged("Properties");
        }

        protected virtual void OnPropertiesChanged(string propertyName = null)
        {
            if (PropertyChanged != null)
                PropertyChanged(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}