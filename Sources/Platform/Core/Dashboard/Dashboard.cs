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
    public abstract class DashboardBase<TProperty> : INotifyPropertyChanged where TProperty : DashboardProperty
    {
        public event PropertyChangedEventHandler PropertyChanged;

        [YAXSerializableField]
        public IList<TProperty> Properties { get; set; }

        public DashboardBase()
        {
            Properties = new List<TProperty>();
        }

        public virtual void RestoreFromIds(MyProject project)
        {
            foreach (var property in Properties.ToList())
            {
                try
                {
                    property.Restore(project);
                }
                catch
                {
                    Remove(property);
                    MyLog.WARNING.WriteLine("A property with identifier \"{0}\" could not be deserialized.",
                        property.PropertyId);
                }
            }
        }

        public void Remove(TProperty property)
        {
            if (!Properties.Remove(property))
                return;

            var memberProperty = property as DashboardNodePropertyBase;
            if (memberProperty != null && memberProperty.Group != null)
                memberProperty.Group.Remove(memberProperty);

            OnPropertiesChanged("Properties");
        }

        public TProperty Get(string propertyId)
        {
            return Properties.FirstOrDefault(p => p.PropertyId == propertyId);
        }

        public abstract void RemoveAll(object target);

        protected void OnPropertiesChanged(string propertyName = null)
        {
            if (PropertyChanged != null)
                PropertyChanged(this, new PropertyChangedEventArgs(propertyName));
        }
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class Dashboard : DashboardBase<DashboardNodePropertyBase>
    {
        public bool Add(object target, string propertyName)
        {
            if (Contains(target, propertyName))
                return false;

            DashboardNodePropertyBase property = DashboardPropertyFactory.CreateProperty(target, propertyName);

            if (property == null)
                throw new InvalidOperationException("Invalid property owner provided");

            if (property.IsReadOnly)
                throw new InvalidOperationException("Readonly properties are not supported");

            Properties.Add(property);
            OnPropertiesChanged("Properties");

            return true;
        }

        public bool Contains(object target, string propertyName)
        {
            return
                Properties.Select(property => property)
                    .Any(property => property.Target == target && property.PropertyName == propertyName);
        }

        public bool Remove(object target, string propertyName)
        {
            DashboardNodePropertyBase property = Properties.FirstOrDefault(p => p.Target == target && p.PropertyName == propertyName);

            if (property == null)
                return false;

            if (Properties.Remove(property))
            {
                if (property.Group != null)
                    property.Group.Remove(property);

                OnPropertiesChanged("Properties");
                return true;
            }

            return false;
        }

        public DashboardNodePropertyBase Get(object target, string propertyName)
        {
            return Properties.FirstOrDefault(p => p.Target == target && p.PropertyName == propertyName);
        }

        public override void RemoveAll(object target)
        {
            List<DashboardNodePropertyBase> toBeRemoved = Properties.Where(property => property.Node == target).ToList();
            foreach (DashboardNodePropertyBase property in toBeRemoved)
            {
                Remove(property);
            }
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

        public void Add()
        {
            var name = "Group " + GetNextId();
            // TODO(HonzaS): Use a temporary set for efficiency?
            while (Properties.Any(property => property.PropertyName == name))
                name = "Group " + GetNextId();

            Properties.Add(new DashboardPropertyGroup(name));
            OnPropertiesChanged("Properties");
        }

        public override void RemoveAll(object target)
        {
            // No need to do anything, the property will remove itself.
        }

        public DashboardPropertyGroup GetByName(string name)
        {
            return Properties.FirstOrDefault(property => property.PropertyName == name);
        }

        /// <summary>
        /// Checks if the given group can set its name to the given parameter.
        /// </summary>
        public bool CanChangeName(DashboardPropertyGroup group, string name)
        {
            return Properties.Where(property => property != group).All(property => property.PropertyName != name);
        }

        public override void RestoreFromIds(MyProject project)
        {
            base.RestoreFromIds(project);

            var groupedByName = Properties.GroupBy(group => group.PropertyName);

            foreach (var grouping in groupedByName)
            {
                var first = true;
                foreach (var property in grouping)
                {
                    if (first)
                    {
                        first = false;
                        continue;
                    }

                    MyLog.WARNING.WriteLine("Multiple properties with the same name: {0}, generating unique name.",
                        grouping.Key);
                    property.PropertyName = Guid.NewGuid().ToString();
                }
            }
        }
    }
}