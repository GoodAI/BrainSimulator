using GoodAI.Modules.School.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.School.GUI
{
    public partial class SchoolAddTaskForm : DockContent
    {
        private struct TypeHolder
        {
            private Type m_type { get; set; }

            public TypeHolder(Type type)
                : this()
            {
                m_type = type;
            }

            public override string ToString()
            {
                return m_type.Name;
            }
        }

        // TODO: will be changed to actual info about type
        public string ResultTask { get; set; }

        public SchoolAddTaskForm()
        {
            InitializeComponent();

            Type taskInterface = typeof(ILearningTask);
            IEnumerable<Type> types = AppDomain.CurrentDomain.GetAssemblies()
                .SelectMany(x => x.GetTypes())
                .Where(x => !x.IsAbstract && x.GetInterfaces().Contains(taskInterface));

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
            // TODO after refactor: obtain supported worlds for task
            comboWorlds.Items.AddRange(new string[] { "PlumberWorld", "RoguelikeWorld" });
        }

        private void btnAdd_Click(object sender, EventArgs e)
        {
            ResultTask = comboTasks.SelectedItem.ToString() + " (" + comboWorlds.SelectedItem.ToString() + ")";
            this.Close();
        }
    }
}
