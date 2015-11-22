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
    public class DashboardBase<TProperty> : INotifyPropertyChanged where TProperty : DashboardProperty
    {
        public event PropertyChangedEventHandler PropertyChanged;

        [YAXSerializableField]
        public IList<TProperty> Properties { get; set; }

        public DashboardBase()
        {
            Properties = new List<TProperty>();
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

        public void Remove(TProperty property)
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

        protected void OnPropertiesChanged(string propertyName = null)
        {
            if (PropertyChanged != null)
                PropertyChanged(this, new PropertyChangedEventArgs(propertyName));
        }
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class Dashboard : DashboardBase<DashboardNodeProperty>
    {
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
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public sealed class GroupDashboard : DashboardBase<DashboardPropertyGroup>
    {
        private static int m_nextId = 1;
        private static int GetNextId()
        {
            return m_nextId++;
        }

        public void AddGroupedProperty()
        {
            var name = "Group " + GetNextId();
            // TODO(HonzaS): Use a temporary set for efficiency?
            while (Properties.Any(property => property.PropertyName == name))
                name = "Group " + GetNextId();

            Properties.Add(new DashboardPropertyGroup
            {
                PropertyName = name
            });
            OnPropertiesChanged("Properties");
        }
    }
}