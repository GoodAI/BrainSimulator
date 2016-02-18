using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;
using GoodAI.BrainSimulator.Forms;
using GoodAI.Modules.School.Common;
using GoodAI.Core.Utils;

namespace GoodAI.School.GUI
{

    [BrainSimUIExtension]
    public partial class LearningTaskSelectionForm : DockContent
    {
        public LearningTaskSelectionForm()
        {
            InitializeComponent();
            learningTaskList.DisplayMember = "DisplayName";
        }

        private void LearningTaskSelectionForm_Load(object sender, EventArgs e)
        {
            PopulateWorldList();
            PopulateLearningTaskList();
        }

        private void PopulateWorldList()
        {
            // TODO 
            throw new NotImplementedException();
        }

        private void PopulateLearningTaskList()
        {
            learningTaskList.Items.Clear();
            foreach (Type type in LearningTaskFactory.KnownLearningTasks.Keys)
            {
                // TODO check if supports world
                learningTaskList.Items.Add(new LearningTaskListItem(type));
            }

            if (learningTaskList.Items.Count > 0)
                learningTaskList.SelectedIndex = 0;
        }

        private void okButton_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.OK;
            Close();
        }

        private void cancelButton_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }

        private void learningTaskList_SelectedValueChanged(object sender, EventArgs e)
        {
            UpdateLearningTaskDescription();
        }

        private void UpdateLearningTaskDescription()
        {
            if (learningTaskList.SelectedItem == null)
            {
                learningTaskDescription.Url = null;
            }
            else
            { 
                const string HTML_DIRECTORY = @"Resources\html";
                string htmlFileName = (learningTaskList.SelectedItem as LearningTaskListItem).HTMLFileName;
                string fullPath = MyResources.GetMyAssemblyPath() + "\\" + HTML_DIRECTORY + "\\" + htmlFileName;
                learningTaskDescription.Navigate(fullPath);
            }
        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void comboWorlds_SelectedIndexChanged(object sender, EventArgs e)
        {
            PopulateLearningTaskList();
        }
    }

    class LearningTaskListItem
    {
        public Type Type { get; set; }

        public LearningTaskListItem(Type type)
        {
            Type = type;
        }

        public String DisplayName
        {
            get
            {
                DisplayNameAttribute attribute = Type.GetCustomAttributes(typeof(DisplayNameAttribute), true).FirstOrDefault() as DisplayNameAttribute;
                return attribute != null ? attribute.DisplayName : "Display name missing!";
            }
        }

        public string HTMLFileName 
        { 
            get
            {
                return Type.Name + ".html";
            }
        }
    }
}
