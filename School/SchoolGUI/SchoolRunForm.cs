using System;
using System.Collections.Generic;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.School.GUI
{
    public partial class SchoolRunForm : DockContent
    {
        public List<LearningTaskNode> Data;
        private BindingSource m_source;

        public SchoolRunForm()
        {
            InitializeComponent();

            dataGridView1.DataSource = Data;
            // using BindingSource is probably better but it wasn't updating; don't know why - postponed
            m_source = new BindingSource();
            m_source.DataSource = Data;
            dataGridView1.DataSource = m_source;
        }

        public void UpdateData()
        {
            m_source.ResetBindings(true);
            dataGridView1.DataSource = Data;
        }

        private void dataGridView1_CellFormatting(object sender, DataGridViewCellFormattingEventArgs e)
        {
            string columnName = dataGridView1.Columns[e.ColumnIndex].Name;
            if (columnName.Equals(TaskType.Name) || columnName.Equals(WorldType.Name))
            {
                // I am not sure about how bad this approach is, but it get things done
                if (e.Value != null)
                {
                    Type typeValue = e.Value as Type;
                    if (typeValue != null)
                        e.Value = typeValue.Name;
                }
            }
        }
    }
}
