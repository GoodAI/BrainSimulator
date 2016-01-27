using System.Collections.Generic;
using System.Windows.Forms;

namespace GoodAI.School.GUI
{
    public partial class SchoolRunForm : Form
    {
        public List<GoodAI.School.GUI.SchoolMainForm.LearningTaskData> Data;
        private BindingSource m_source;

        public SchoolRunForm()
        {
            InitializeComponent();

            m_source = new BindingSource();
            m_source.DataSource = Data;
            dataGridView1.DataSource = m_source;
        }

        public void Update()
        {
            m_source.ResetBindings(false);
        }
    }
}
