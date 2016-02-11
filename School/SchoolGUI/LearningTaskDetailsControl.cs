using System;
using System.Linq;
using System.ComponentModel;
using System.Windows.Forms;

namespace GoodAI.School.GUI
{
    public partial class LearningTaskDetailsControl : UserControl
    {
        private Type m_taskType;
        public Type TaskType
        {
            get { return m_taskType; }
            set
            {
                m_taskType = value;
                if (m_taskType != null)
                    labelDescriptionControl1.Type = m_taskType;
            }
        }

        public LearningTaskDetailsControl()
        {
            InitializeComponent();
        }
    }
}
