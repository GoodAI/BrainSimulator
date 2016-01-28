using System.Collections.Generic;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.School.GUI
{
    public partial class SchoolRunForm : DockContent
    {
        public List<GoodAI.School.GUI.SchoolMainForm.LearningTaskData> Data;
        private BindingSource m_source;

        public SchoolRunForm()
        {
            InitializeComponent();

            dataGridView1.DataSource = Data;
            // using BindingSource is probably better but it wasn't updating; don't know why - postponed
            //m_source = new BindingSource();
            //m_source.DataSource = Data;
            //dataGridView1.DataSource = m_source;
        }

        public void UpdateData()
        {
            // m_source.ResetBindings(true);
            dataGridView1.DataSource = Data;
        }
    }
}
