using System;
using System.Windows.Forms;

namespace GoodAI.School.GUI
{
    public partial class LearningTaskDetailsControl : UserControl
    {
        private readonly Type m_taskType;

        public LearningTaskDetailsControl() : this(null) { }

        public LearningTaskDetailsControl(Type TaskType)
        {
            m_taskType = TaskType;
            InitializeComponent();

            //if(m_taskType.IsAssignableFrom())
            //string a =
        }
    }
}
