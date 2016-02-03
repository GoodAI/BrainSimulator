using Aga.Controls.Tree;
using Aga.Controls.Tree.NodeControls;
using GoodAI.BrainSimulator.Forms;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
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
        private string m_lastOpenedFile;

        public SchoolMainForm()
        {
            m_serializer = new YAXSerializer(typeof(PlanDesign));
            AddTaskView = new SchoolAddTaskForm();
            RunView = new SchoolRunForm();

            InitializeComponent();

            m_model = new TreeModel();
            tree.Model = m_model;
            tree.Refresh();

            checkBoxAutosave.Checked = Properties.School.Default.AutosaveEnabled;
            m_lastOpenedFile = Properties.School.Default.LastOpenedFile;
            if (LoadCurriculum(m_lastOpenedFile))
                saveFileDialog1.FileName = m_lastOpenedFile;

            UpdateButtons();
        }

        private void SchoolMainForm_Load(object sender, System.EventArgs e) { }

        #region Curricula

        private CurriculumNode AddCurriculum()
        {
            CurriculumNode node = new CurriculumNode();
            node.Text = "Curr" + m_model.Nodes.Count.ToString();
            m_model.Nodes.Add(node);
            return node;
        }

        private bool LoadCurriculum(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                return false;

            string xmlCurr;
            try { xmlCurr = File.ReadAllText(filePath); }
            catch (IOException e)
            {
                MyLog.WARNING.WriteLine("Unable to read file " + filePath);
                return false;
            }

            try
            {
                PlanDesign plan = (PlanDesign)m_serializer.Deserialize(xmlCurr);
                List<CurriculumNode> currs = (List<CurriculumNode>)plan;

                foreach (CurriculumNode curr in currs)
                    m_model.Nodes.Add(curr);
            }
            catch (YAXException e)
            {
                MyLog.WARNING.WriteLine("Unable to deserialize data from " + filePath);
                return false;
            }

            Properties.School.Default.LastOpenedFile = filePath;
            Properties.School.Default.Save();

            return false;
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

        private void SetButtonsEnabled(Control control, bool value)
        {
            Action<Control> setBtns = (x) =>
            {
                Button b = x as Button;
                if (b != null)
                    b.Enabled = value;
            };
            ApplyToAll(control, setBtns);
        }

        private void DisableButtons(Control control)
        {
            SetButtonsEnabled(control, false);
        }

        private void EnableButtons(Control control)
        {
            SetButtonsEnabled(control, true);
        }

        private void UpdateButtons()
        {
            EnableButtons(this);

            if (!tree.AllNodes.Any())
                btnSave.Enabled = btnSaveAs.Enabled = btnRun.Enabled = false;

            if (tree.SelectedNode == null)
            {
                btnDeleteCurr.Enabled = btnDetailsCurr.Enabled = false;
                DisableButtons(groupBoxTask);
                return;
            }

            SchoolTreeNode selected = tree.SelectedNode.Tag as SchoolTreeNode;
            Debug.Assert(selected != null);

            if (selected is CurriculumNode)
                btnDeleteTask.Enabled = btnDetailsTask.Enabled = false;

            // to be removed
            btnDetailsCurr.Enabled = btnDetailsTask.Enabled = false;
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

        private void btnDeleteCurr_Click(object sender, EventArgs e)
        {
            if (tree.SelectedNode.Tag is CurriculumNode)
            {
                DeleteNode(sender, e);
                return;
            }
            Node parent = (tree.SelectedNode.Tag as Node).Parent;
            if (parent != null && parent is CurriculumNode)
                parent.Parent = null;
        }

        private void btnNewTask_Click(object sender, EventArgs e)
        {
            if (tree.SelectedNode == null)
                return;

            AddTaskView.ShowDialog(this);
            if (AddTaskView.ResultTask == null)
                return;

            LearningTaskNode task = new LearningTaskNode(AddTaskView.ResultTaskType, AddTaskView.ResultWorldType);
            if (tree.SelectedNode.Tag is CurriculumNode)
            {
                (tree.SelectedNode.Tag as Node).Nodes.Add(task);
                tree.SelectedNode.IsExpanded = true;
            }
            else if (tree.SelectedNode.Tag is LearningTaskNode)
            {
                LearningTaskNode source = tree.SelectedNode.Tag as LearningTaskNode;
                int targetPositon = source.Parent.Nodes.IndexOf(source);
                source.Parent.Nodes.Insert(targetPositon + 1, task);
            }
        }

        private void DeleteNode(object sender, EventArgs e)
        {
            (tree.SelectedNode.Tag as Node).Parent = null;
        }

        private void checkBoxAutosave_CheckedChanged(object sender, EventArgs e)
        {
            Properties.School.Default.AutosaveEnabled = checkBoxAutosave.Checked;
            Properties.School.Default.Save();
        }

        private void btnRun_Click(object sender, EventArgs e)
        {
            OpenFloatingOrActivate(RunView, DockPanel);
            List<LearningTaskNode> data = new List<LearningTaskNode>();

            //PlanDesign design = new PlanDesign(m_model.Nodes.Where(x => x is CurriculumNode).Select(x => x as CurriculumNode));
            IEnumerable<CurriculumNode> activeCurricula = m_model.Nodes.
                Where(x => x is CurriculumNode).
                Select(x => x as CurriculumNode).
                Where(x => x.Enabled == true);

            IEnumerable<LearningTaskNode> ltNodes = activeCurricula.
                SelectMany(x => (x as CurriculumNode).Nodes).
                Select(x => x as LearningTaskNode).
                Where(x => x.Enabled == true);

            foreach (LearningTaskNode ltNode in ltNodes)
                data.Add(ltNode);
            RunView.Data = data;
            RunView.UpdateData();
        }

        private bool AddFileContent(bool clearWorkspace = false)
        {
            if (openFileDialog1.ShowDialog() != DialogResult.OK)
                return false;
            if (clearWorkspace)
                m_model.Nodes.Clear();
            LoadCurriculum(openFileDialog1.FileName);
            return true;
        }

        private void btnOpen_Click(object sender, EventArgs e)
        {
            AddFileContent(true);
        }

        private void btnSave_Click(object sender, EventArgs e)
        {
            if (saveFileDialog1.FileName != string.Empty)
                SaveProject(saveFileDialog1.FileName);
            else
                SaveProjectAs(sender, e);  // ask for file name and then save the project
        }

        private void btnImport_Click(object sender, EventArgs e)
        {
            AddFileContent();
        }

        #endregion

        #region (De)serialization

        private void SaveProject(string path)
        {
            PlanDesign plan = new PlanDesign(m_model.Nodes.Where(x => x is CurriculumNode).Select(x => x as CurriculumNode).ToList());

            string xmlResult = m_serializer.Serialize(plan);
            File.WriteAllText(path, xmlResult);
        }

        private void SaveProjectAs(object sender, EventArgs e)
        {
            if (saveFileDialog1.ShowDialog() != DialogResult.OK)
                return;

            SaveProject(saveFileDialog1.FileName);
            Properties.School.Default.LastOpenedFile = saveFileDialog1.FileName;
            Properties.School.Default.Save();
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

        private void SchoolMainForm_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Delete)
                DeleteNode(sender, null);
        }
    }
}
