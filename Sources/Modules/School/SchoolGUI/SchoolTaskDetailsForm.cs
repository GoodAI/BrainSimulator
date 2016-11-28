using System;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.School.GUI
{
    public partial class SchoolTaskDetailsForm : DockContent
    {
        public SchoolTaskDetailsForm(Type taskType)
        {
            InitializeComponent();
            this.learningTaskDescription.LearningTaskType = taskType;
        }
    }
}
