using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;
using GoodAI.Core.Utils;
using GoodAI.School.Common;

namespace GoodAI.School.GUI
{
    public partial class LearningTaskSelectionForm : DockContent
    {
        /// <summary>
        /// In order to control itemcheck changes (blinds double clicking, among other things)
        /// </summary>
        bool AuthorizeCheck { get; set; }

        public LearningTaskSelectionForm()
        {
            InitializeComponent();
            learningTaskList.DisplayMember = "DisplayName";
            worldList.DisplayMember = "DisplayName";
        }

        public List<Type> ResultLearningTaskTypes { get; set; }
        public Type ResultWorldType { get; set; }

        private void LearningTaskSelectionForm_Load(object sender, EventArgs e)
        {
            PopulateWorldList();
            PopulateLearningTaskList();
        }

        private void PopulateWorldList()
        {
            foreach (Type type in CurriculumManager.GetAvailableWorlds())
            {
                worldList.Items.Add(new TypeListItem(type));
            }
            if (worldList.Items.Count > 0)
                worldList.SelectedIndex = 0;
        }

        private void PopulateLearningTaskList()
        {
            learningTaskList.Items.Clear();
            Type selectedWorldType = (worldList.SelectedItem as TypeListItem).Type;

            foreach (Type entry in CurriculumManager.GetTasksForWorld(selectedWorldType))
                learningTaskList.Items.Add(new LearningTaskListItem(entry));

            if (learningTaskList.Items.Count > 0)
                learningTaskList.SelectedIndex = 0;
        }

        private void okButton_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.OK;
            ResultLearningTaskTypes = new List<Type>();
            foreach (var item in learningTaskList.CheckedItems)
            {
                ResultLearningTaskTypes.Add((item as TypeListItem).Type);
            }
            ResultWorldType = (worldList.SelectedItem as TypeListItem).Type;
            Close();
        }

        private void cancelButton_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            ResultLearningTaskTypes = null;
            ResultWorldType = null;
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

        private void worldList_SelectedIndexChanged(object sender, EventArgs e)
        {
            PopulateLearningTaskList();
        }

        // Implements "check only when box clicked" behavior in the checkedlistbox
        // See http://stackoverflow.com/questions/2093961/checkedlistbox-control-only-checking-the-checkbox-when-the-actual-checkbox-is
        private void learningTaskList_MouseClick(object sender, MouseEventArgs e)
        {
            const int CHECK_BOX_WIDTH = 16;
            for (int i = 0; i < this.learningTaskList.Items.Count; i++)
            {
                Rectangle rec = this.learningTaskList.GetItemRectangle(i);
                rec.Width = CHECK_BOX_WIDTH;

                if (rec.Contains(e.Location))
                {
                    AuthorizeCheck = true;
                    bool newValue = !this.learningTaskList.GetItemChecked(i);
                    this.learningTaskList.SetItemChecked(i, newValue);
                    AuthorizeCheck = false;

                    return;
                }
            }
        }

        // Implements "check only when box clicked" behavior in the checkedlistbox
        private void learningTaskList_ItemCheck(object sender, ItemCheckEventArgs e)
        {
            if (!AuthorizeCheck)
                e.NewValue = e.CurrentValue; //check state change was not through authorized actions
        }
    }

    class TypeListItem
    {
        public Type Type { get; set; }

        public TypeListItem(Type type)
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
    }

    class LearningTaskListItem : TypeListItem
    {
        public LearningTaskListItem(Type type)
            : base(type)
        {
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
