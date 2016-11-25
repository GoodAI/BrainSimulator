using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.School.GUI
{
    public partial class SchoolCurrDetailsForm : DockContent
    {
        private readonly CurriculumNode m_node;

        public SchoolCurrDetailsForm(CurriculumNode node)
        {
            m_node = node;
            InitializeComponent();
            textBox1.Text = node.Description;
        }

        private void btnSave_Click(object sender, System.EventArgs e)
        {
            m_node.Description = textBox1.Text;
        }

        private void btnClose_Click(object sender, System.EventArgs e)
        {
            Close();
        }
    }
}
