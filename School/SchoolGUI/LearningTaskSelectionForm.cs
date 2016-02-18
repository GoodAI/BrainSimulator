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
using GoodAI.Modules.School.Worlds;

namespace GoodAI.School.GUI
{
    [BrainSimUIExtension]
    public partial class LearningTaskSelectionForm : DockContent
    {
        public LearningTaskSelectionForm()
        {
            InitializeComponent();
            learningTaskList.DisplayMember = "DisplayName";
            worldList.DisplayMember = "DisplayName";
        }

        private void LearningTaskSelectionForm_Load(object sender, EventArgs e)
        {
            PopulateWorldList();
            PopulateLearningTaskList();
        }

        private void PopulateWorldList()
        {
            var interfaceType = typeof(IWorldAdapter);
            var types = AppDomain.CurrentDomain.GetAssemblies()
                .SelectMany(s => s.GetTypes())
                .Where(p => interfaceType.IsAssignableFrom(p) && !p.IsInterface && !p.IsAbstract);
            foreach (Type type in types)
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
            foreach (var entry in LearningTaskFactory.KnownLearningTasks)
            {
                Type learningTaskType = entry.Key;
                List<Type> worldTypes = entry.Value;
                if (ContainsType(worldTypes, selectedWorldType))
                if (worldTypes.Contains(selectedWorldType))
                { 
                    learningTaskList.Items.Add(new LearningTaskListItem(learningTaskType));
                }
            }

            if (learningTaskList.Items.Count > 0)
                learningTaskList.SelectedIndex = 0;
        }

        private bool ContainsType(List<Type> worldTypes, Type selectedWorldType)
        {
            foreach (Type type in worldTypes)
            {
                if (selectedWorldType == type || selectedWorldType.IsSubclassOf(type))
                {
                    return true;
                }
            }
            return false;
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

        private void worldList_SelectedIndexChanged(object sender, EventArgs e)
        {
            PopulateLearningTaskList();
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
        public LearningTaskListItem(Type type) : base(type)
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
