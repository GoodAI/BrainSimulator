using System;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.School.GUI
{
    public partial class SchoolTaskDetailsForm : DockContent
    {
        private readonly Type m_taskType;

        public SchoolTaskDetailsForm(Type taskType)
        {
            m_taskType = taskType;
            InitializeComponent();

            //small hack - thanks to this, user control will be initialized with TaskType AND you are able to see this form's designer
            this.learningTaskDetailsControl1 = new GoodAI.School.GUI.LearningTaskDetailsControl(m_taskType);
        }
    }
}
