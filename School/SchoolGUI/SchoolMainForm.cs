using Aga.Controls.Tree;
using Aga.Controls.Tree.NodeControls;
using GoodAI.BrainSimulator.Forms;
using GoodAI.Modules.School.Common;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;
using YAXLib;

namespace GoodAI.School.GUI
{
    [BrainSimUIExtension]
    public partial class SchoolMainForm : DockContent
    {
        public SchoolAddTaskForm AddTaskView { get; private set; }
        public SchoolRunForm RunView { get; private set; }
        private YAXSerializer m_serializer;
        private TreeModel m_model;
        private string m_curriculaPath;

        #region Helper classes

        public class SchoolTreeNode : Node
        {
            public SchoolTreeNode(string text) : base(text) { }
        }

        public class CurriculumNode : SchoolTreeNode
        {
            public CurriculumNode(string text) : base(text) { }
        }

        public class LearningTaskNode : SchoolTreeNode
        {
            public bool Enabled { get; set; }
            public Type Type { get; set; }
            public Type WorldType { get; set; }
            public LearningTaskNode(string text) : base(text) { }
            public LearningTaskNode(Type taskType) : base(taskType.Name) { }
        }

        public class LearningTaskData
        {
            private ILearningTask m_task;

            public string Name
            {
                get
                {
                    return m_task.GetType().Name;
                }
            }

            public string WorldName { get; private set; }
            public int Steps { get; set; }
            public float Time { get; set; }
            public string Status { get; set; }

            public LearningTaskData(ILearningTask task)
            {
                m_task = task;
            }
        }

        #endregion

        public SchoolMainForm()
        {
            m_serializer = new YAXSerializer(typeof(SchoolCurriculum));
            AddTaskView = new SchoolAddTaskForm();
            RunView = new SchoolRunForm();

            InitializeComponent();

            m_model = new TreeModel();
            tree.Model = m_model;
            tree.Refresh();
            UpdateButtons();

            checkBoxAutosave.Checked = Properties.School.Default.AutosaveEnabled;
            m_curriculaPath = Properties.School.Default.CurriculaFolder;
            ReloadCurricula();
        }

        private void SchoolMainForm_Load(object sender, System.EventArgs e) { }

        #region Curricula

        private CurriculumNode AddCurriculum()
        {
            CurriculumNode node = new CurriculumNode("Curr" + m_model.Nodes.Count.ToString());
            m_model.Nodes.Add(node);
            return node;
        }

        private void ReloadCurricula()
        {
            m_model.Nodes.Clear();

            if (String.IsNullOrEmpty(m_curriculaPath))
                return;

            foreach (string filePath in Directory.GetFiles(m_curriculaPath))
            {
                LoadCurriculum(filePath);
            }
        }

        private void LoadCurriculum(string filePath)
        {
            string xmlCurr = File.ReadAllText(filePath);

            try
            {
                SchoolCurriculum curr = (SchoolCurriculum)m_serializer.Deserialize(xmlCurr);
                CurriculumNode node = CurriculumDataToCurriculumNode(curr);
                m_model.Nodes.Add(node);
            }
            catch (YAXException) { }
        }

        private SchoolCurriculum CurriculumNodeToCurriculumData(CurriculumNode node)
        {
            SchoolCurriculum data = new SchoolCurriculum();
            data.Name = node.Text;

            foreach (LearningTaskNode taskNode in node.Nodes)
                data.AddLearningTask(taskNode.Type, taskNode.WorldType);

            return data;
        }

        private CurriculumNode CurriculumDataToCurriculumNode(SchoolCurriculum data)
        {
            CurriculumNode node = new CurriculumNode(data.Name);

            foreach (ILearningTask task in data)
            {
                // TODO: World name can be displayed through reflection OR once World param is in ILearningTask (or SchoolCurriculum is restricted to AbstractLTs)
                LearningTaskNode taskNode = new LearningTaskNode(task.GetType());
                node.Nodes.Add(taskNode);
            }

            return node;
        }

        private List<LearningTaskData> CurriculumDataToLTData(SchoolCurriculum curriculum)
        {
            List<LearningTaskData> result = new List<LearningTaskData>();
            foreach (ILearningTask task in curriculum)
            {
                LearningTaskData data = new LearningTaskData(task);
                result.Add(data);
            }

            return result;
        }

        #endregion

        #region UI

        private void ApplyToAll(Control parent, Action<Control> apply)
        {
            foreach (Control control in parent.Controls)
            {
                if (control.HasChildren)
                    ApplyToAll(control, apply);
                apply(control);
            }
        }

        private void HideAllButtons()
        {
            Action<Control> hideBtns = (x) =>
            {
                Button b = x as Button;
                if (b != null)
                    b.Visible = false;
            };
            ApplyToAll(this, hideBtns);
        }

        private void UpdateButtons()
        {
            HideAllButtons();
            btnNewCurr.Visible = btnImportCurr.Visible = btnCurrFolder.Visible = true;
            if (tree.SelectedNode == null)
                return;

            SchoolTreeNode selected = tree.SelectedNode.Tag as SchoolTreeNode;
            Debug.Assert(selected != null);

            if (selected is CurriculumNode)
                groupBox1.Text = "Curriculum";
            else if (selected is LearningTaskNode)
                groupBox1.Text = "Learning task";

            btnDelete.Visible = true;
            btnDetailsCurr.Enabled = btnDetailsTask.Enabled = false;
            btnNewCurr.Visible = btnDetailsCurr.Visible = btnExportCurr.Visible = btnImportCurr.Visible = btnNewTask.Visible = btnRun.Visible =
                selected is CurriculumNode;
            btnDetailsTask.Visible = selected is LearningTaskNode;
        }

        #endregion

        #region DragDrop

        private void tree_ItemDrag(object sender, System.Windows.Forms.ItemDragEventArgs e)
        {
            tree.DoDragDropSelectedNodes(DragDropEffects.Move);
        }

        private void tree_DragOver(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(typeof(TreeNodeAdv[])) && tree.DropPosition.Node != null)
            {
                TreeNodeAdv[] nodes = e.Data.GetData(typeof(TreeNodeAdv[])) as TreeNodeAdv[];
                TreeNodeAdv parent = tree.DropPosition.Node;
                if (tree.DropPosition.Position != NodePosition.Inside)
                    parent = parent.Parent;

                foreach (TreeNodeAdv node in nodes)
                    if (!CheckNodeParent(parent, node))
                    {
                        e.Effect = DragDropEffects.None;
                        return;
                    }

                e.Effect = e.AllowedEffect;
            }
        }

        private bool CheckNodeParent(TreeNodeAdv parent, TreeNodeAdv node)
        {
            while (parent != null)
            {
                if (node == parent)
                    return false;
                else
                    parent = parent.Parent;
            }
            return true;
        }

        private void tree_DragDrop(object sender, DragEventArgs e)
        {
            tree.BeginUpdate();

            TreeNodeAdv[] nodes = (TreeNodeAdv[])e.Data.GetData(typeof(TreeNodeAdv[]));
            Node dropNode = tree.DropPosition.Node.Tag as Node;
            if (tree.DropPosition.Position == NodePosition.Inside)
            {
                foreach (TreeNodeAdv n in nodes)
                {
                    (n.Tag as Node).Parent = dropNode;
                }
                tree.DropPosition.Node.IsExpanded = true;
            }
            else
            {
                Node parent = dropNode.Parent;
                Node nextItem = dropNode;
                if (tree.DropPosition.Position == NodePosition.After)
                    nextItem = dropNode.NextNode;

                foreach (TreeNodeAdv node in nodes)
                    (node.Tag as Node).Parent = null;

                int index = -1;
                index = parent.Nodes.IndexOf(nextItem);
                foreach (TreeNodeAdv node in nodes)
                {
                    Node item = node.Tag as Node;
                    if (index == -1)
                        parent.Nodes.Add(item);
                    else
                    {
                        parent.Nodes.Insert(index, item);
                        index++;
                    }
                }
            }

            tree.EndUpdate();
        }

        #endregion

        #region Button clicks

        private void btnNewCurr_Click(object sender, EventArgs e)
        {
            CurriculumNode newCurr = AddCurriculum();
        }

        private void btnNewTask_Click(object sender, EventArgs e)
        {
            if (tree.SelectedNode == null || !(tree.SelectedNode.Tag is CurriculumNode))
                return;

            AddTaskView.ShowDialog(this);
            if (AddTaskView.ResultTask == null)
                return;

            LearningTaskNode newTask = new LearningTaskNode(AddTaskView.ResultTask);
            newTask.Type = AddTaskView.ResultTaskType;
            newTask.WorldType = AddTaskView.ResultWorldType;
            (tree.SelectedNode.Tag as Node).Nodes.Add(newTask);
            tree.SelectedNode.IsExpanded = true;
        }

        private void btnDelete_Click(object sender, EventArgs e)
        {
            if (tree.SelectedNode != null && tree.SelectedNode.Tag is Node)
                (tree.SelectedNode.Tag as Node).Parent = null;
        }

        private void btnExportCurr_Click(object sender, EventArgs e)
        {
            SchoolCurriculum test = CurriculumNodeToCurriculumData(tree.SelectedNode.Tag as CurriculumNode);
            string xmlCurr = m_serializer.Serialize(test);
            saveFileDialog1.ShowDialog();
            File.WriteAllText(saveFileDialog1.FileName, xmlCurr);
        }

        private void btnImportCurr_Click(object sender, EventArgs e)
        {
            openFileDialog1.ShowDialog();
            LoadCurriculum(openFileDialog1.FileName);
        }

        private void checkBoxAutosave_CheckedChanged(object sender, EventArgs e)
        {
            Properties.School.Default.AutosaveEnabled = checkBoxAutosave.Checked;
            Properties.School.Default.Save();
        }

        private void btnCurrFolder_Click(object sender, EventArgs e)
        {
            folderBrowserDialog1.ShowDialog();
            m_curriculaPath = folderBrowserDialog1.SelectedPath;
            Properties.School.Default.CurriculaFolder = m_curriculaPath;
            Properties.School.Default.Save();
            ReloadCurricula();
        }

        private void btnRun_Click(object sender, EventArgs e)
        {
            OpenFloatingOrActivate(RunView, DockPanel);
            SchoolCurriculum curriculum = CurriculumNodeToCurriculumData(tree.SelectedNode.Tag as CurriculumNode);
            List<LearningTaskData> data = CurriculumDataToLTData(curriculum);
            RunView.Data = data;
            RunView.UpdateData();
        }

        #endregion

        // almost same as Mainform.OpenFloatingOrActivate - refactor?
        private void OpenFloatingOrActivate(DockContent view, DockPanel panel)
        {
            if ((view.DockAreas & DockAreas.Float) > 0 && !view.Created)
            {
                Size viewSize = new Size(view.Bounds.Size.Width, view.Bounds.Size.Height);
                view.Show(panel, DockState.Float);
                view.FloatPane.FloatWindow.Size = viewSize;
            }
            else
            {
                view.Activate();
            }
        }

        private void tree_SelectionChanged(object sender, EventArgs e)
        {
            UpdateButtons();
        }

        private void nodeTextBox1_DrawText(object sender, DrawTextEventArgs e)
        {
            if (e.Node.IsSelected)
                e.Font = new System.Drawing.Font(e.Font, FontStyle.Bold);
        }
    }
}
