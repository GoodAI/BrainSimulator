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

        // Tuple is <TaskType, WorldType>
        public HashSet<Tuple<Type,Type>> ResultLearningTaskTypes { get; set; }
        private Type m_currentWorldType { get; set; }

        private void LearningTaskSelectionForm_Load(object sender, EventArgs e)
        {
            ResultLearningTaskTypes = new HashSet<Tuple<Type,Type>>();
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

            List<Type> worldTasks = CurriculumManager.GetTasksForWorld(selectedWorldType);
            List<Type> checkedTasks = ResultLearningTaskTypes.Select(resultLearningTaskType => resultLearningTaskType.Item1).ToList();

            AuthorizeCheck = true;

            for (int i = 0; i < worldTasks.Count; i++)
            {
                learningTaskList.Items.Add(new LearningTaskListItem(worldTasks[i]));
                learningTaskList.SetItemChecked(i, checkedTasks.Contains(worldTasks[i]));
            }
            AuthorizeCheck = false;

            if (learningTaskList.Items.Count > 0)
                learningTaskList.SelectedIndex = 0;
        }

        private void CollectCheckedLTs()
        {
            for (int i = 0; i < learningTaskList.Items.Count; i++)
            {
                TypeListItem typeListItem = learningTaskList.Items[i] as TypeListItem;
                if (typeListItem == null) continue;

                Type ltType = typeListItem.Type;
                Type worldType = m_currentWorldType;    // this is the value of last known world (it is not reflecting the real "current" at this place
                if (learningTaskList.GetItemCheckState(i) == CheckState.Checked)
                    ResultLearningTaskTypes.Add(new Tuple<Type, Type>(ltType, worldType));
                else
                    ResultLearningTaskTypes.Remove(new Tuple<Type, Type>(ltType, worldType));
            }
        }

        private void okButton_Click(object sender, EventArgs e)
        {
            CollectCheckedLTs();
            DialogResult = DialogResult.OK;
            Close();
        }

        private void cancelButton_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            ResultLearningTaskTypes = null;
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
            CollectCheckedLTs();
            PopulateLearningTaskList();
            // this assignement is here so that CollectCheckedLTs still uses the last known world
            m_currentWorldType = (worldList.SelectedItem as TypeListItem).Type;
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
