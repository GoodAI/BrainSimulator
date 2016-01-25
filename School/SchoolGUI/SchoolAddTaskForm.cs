using GoodAI.Modules.School.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.School.GUI
{
    public partial class SchoolAddTaskForm : DockContent
    {
        // this is here just for overriding ToString (for combobox list) - ask Martin P. if whole type's ToString is OK - if yes, delete this class
        private class TypeHolder
        {
            public Type Type { get; set; }

            public TypeHolder(Type type)
            {
                Type = type;
            }

            public override string ToString()
            {
                return Type.Name;
            }
        }

        public string ResultTask { get; set; }  // TODO:extract name from task type
        public Type ResultTaskType { get; set; }
        public Type ResultWorldType { get; set; }

        public SchoolAddTaskForm()
        {
            InitializeComponent();

            Type taskType = typeof(AbstractLearningTask<>);
            //TODO: check multi-level inheritance if there will be any in future
            //TODO: check only some assemblies
            IEnumerable<Type> types = AppDomain.CurrentDomain.GetAssemblies()
                .SelectMany(x => x.GetTypes())
                .Where(x => x.BaseType != null && x.BaseType.IsGenericType && x.BaseType.GetGenericTypeDefinition() == taskType);

            foreach (Type type in types)
            {
                TypeHolder th = new TypeHolder(type);
                comboTasks.Items.Add(th);
            }

            if (comboTasks.Items.Count > 0)
                comboTasks.SelectedIndex = 0;
            if (comboWorlds.Items.Count > 0)
                comboWorlds.SelectedIndex = 0;

            this.AcceptButton = btnAdd;
        }

        private List<Type> GetSupportedWorlds(Type taskType)
        {
            //TODO: check multi-level inheritance if there will be any in future
            // get generic parameter
            Type genericType = taskType.BaseType.GetGenericArguments()[0];
            // look up all derived classes of this type
            //TODO: check only some assemblies
            List<Type> results = AppDomain.CurrentDomain.GetAssemblies()
                .SelectMany(x => x.GetTypes())
                .Where(x => x.BaseType == genericType)
                .ToList();
            results.Add(genericType);
            // remove abstract classes
            results = results.Where(x => !x.IsAbstract).ToList();

            return results;
        }

        private void comboTasks_SelectedIndexChanged(object sender, EventArgs e)
        {
            comboWorlds.Items.Clear();
            List<Type> worlds = GetSupportedWorlds((comboTasks.SelectedItem as TypeHolder).Type);
            foreach (Type world in worlds)
            {
                TypeHolder th = new TypeHolder(world);
                comboWorlds.Items.Add(th);
            }
        }

        private void btnAdd_Click(object sender, EventArgs e)
        {
            ResultTask = comboTasks.SelectedItem.ToString() + " (" + comboWorlds.SelectedItem.ToString() + ")";
            ResultTaskType = (comboTasks.SelectedItem as TypeHolder).Type;
            //ResultWorldType = (comboWorlds.SelectedItem as TypeHolder).Type;
            this.Close();
        }
    }
}
