using System;
using System.Linq;
using WeifenLuo.WinFormsUI.Docking;

namespace SchoolGUI
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

            //Type type = typeof(AbstractLearningTask);
            string typeName = "AbstractLearningTask";
            var types = AppDomain.CurrentDomain.GetAssemblies()
                .SelectMany(x => x.GetTypes())
                .Where(x => x.BaseType != null && x.BaseType.Name == typeName && x.Name != typeName);
            //.Where(x => type.IsAssignableFrom(x) && x != type);

            foreach (var one in types)
            {
                TypeHolder th = new TypeHolder(one);
                comboTasks.Items.Add(th);
            }

            comboTasks.SelectedIndex = 0;
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
