using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;

namespace GoodAI.School.GUI
{
    public partial class LearningTaskBrowser : WebBrowser
    {
        private Type m_learningTaskType; 

        public Type LearningTaskType
        {
            get
            {
                return m_learningTaskType;
            }

            set
            {
                m_learningTaskType = value;
                if (HasLearningTask)
                {
                    Navigate(FullPath);
                }
            }
        }

        public bool HasLearningTask
        {
            get
            {
                return LearningTaskType != null && typeof(ILearningTask).IsAssignableFrom(LearningTaskType);
            }
        }

        public string HTMLFileName
        {
            get
            {
                return HasLearningTask ? LearningTaskType.Name + ".html" : null;
            }
        }

        public string FullPath
        {
            get
            {
                const string HTML_DIRECTORY = @"Resources\html";
                return HasLearningTask ? MyResources.GetMyAssemblyPath() + "\\" + HTML_DIRECTORY + "\\" + HTMLFileName : null;
            }
        }

        public LearningTaskBrowser()
        {
            InitializeComponent();
        }

        protected override void OnPaint(PaintEventArgs pe)
        {
            base.OnPaint(pe);
        }

    }
}
