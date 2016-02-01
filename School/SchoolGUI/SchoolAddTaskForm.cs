using GoodAI.Modules.School.Common;
using System;
using System.Collections.Generic;
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

            IEnumerable<Type> types = LearningTaskFactory.KnownLearningTasks.Keys;

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

        private void comboTasks_SelectedIndexChanged(object sender, EventArgs e)
        {
            comboWorlds.Items.Clear();
            List<Type> worlds = LearningTaskFactory.GetSupportedWorlds((comboTasks.SelectedItem as TypeHolder).Type);
            foreach (Type world in worlds)
            {
                TypeHolder th = new TypeHolder(world);
                comboWorlds.Items.Add(th);
            }

            if (comboWorlds.Items.Count > 0)
                comboWorlds.SelectedIndex = 0;
        }

        private void btnAdd_Click(object sender, EventArgs e)
        {
            ResultTask = comboTasks.SelectedItem.ToString() + " (" + comboWorlds.SelectedItem.ToString() + ")";
            ResultTaskType = (comboTasks.SelectedItem as TypeHolder).Type;
            ResultWorldType = (comboWorlds.SelectedItem as TypeHolder).Type;
            this.Close();
        }
    }
}
