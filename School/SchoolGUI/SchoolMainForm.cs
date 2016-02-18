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
        private const string DEFAULT_FORM_NAME = "School for AI";

        private readonly MainForm m_mainForm;
        private YAXSerializer m_serializer;
        private TreeModel m_model;
        private string m_lastOpenedFile;
        private string m_savedRepresentation;
        private string m_currentFile;

        public LearningTaskSelectionForm AddTaskView { get; private set; }
        public SchoolRunForm RunView { get; private set; }

        private PlanDesign m_design
        {
            get
            {
                return new PlanDesign(m_model.Nodes.Where(x => x is CurriculumNode).Select(x => x as CurriculumNode).ToList());
            }
        }

        public SchoolMainForm(MainForm mainForm)
        {
            m_serializer = new YAXSerializer(typeof(PlanDesign));
            m_mainForm = mainForm;
            RunView = new SchoolRunForm(m_mainForm);

            InitializeComponent();

            m_model = new TreeModel();
            tree.Model = m_model;
            tree.Refresh();

            btnAutosave.Checked = Properties.School.Default.AutosaveEnabled;
            btnAutorun.Checked = Properties.School.Default.AutorunEnabled;
            m_lastOpenedFile = Properties.School.Default.LastOpenedFile;
            if (LoadCurriculum(m_lastOpenedFile))
                saveFileDialog1.FileName = m_lastOpenedFile;

            UpdateButtons();
        }

        private void SchoolMainForm_Load(object sender, System.EventArgs e) { }

        private void UpdateWindowName(object sender, EventArgs e)
        {
            if (!Visible)
            {
                Text = DEFAULT_FORM_NAME;
                return;
            }
            string lof = Properties.School.Default.LastOpenedFile;
            string filename = String.IsNullOrEmpty(Properties.School.Default.LastOpenedFile) ? "Unsaved workspace" : Path.GetFileName(Properties.School.Default.LastOpenedFile);
            Text = DEFAULT_FORM_NAME + " - " + filename;
            if (!IsWorkspaceSaved())
                Text += '*';
        }

        private bool IsWorkspaceSaved()
        {
            if (m_savedRepresentation == null)
                return false;
            string currentRepresentation = m_serializer.Serialize(m_design);
            return m_savedRepresentation.Equals(currentRepresentation);
        }

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

        private void SetToolstripButtonsEnabled(Control control, bool value)
        {
            ToolStrip tools = control as ToolStrip;
            if (tools != null)
                foreach (ToolStripItem item in tools.Items)
                    if (item as ToolStripButton != null)
                        item.Enabled = value;
        }

        private void DisableButtons(Control control)
        {
            SetButtonsEnabled(control, false);
            SetToolstripButtonsEnabled(control, false);
        }

        private void EnableButtons(Control control)
        {
            SetButtonsEnabled(control, true);
        }

        private void EnableToolstripButtons(ToolStrip toolstrip)
        {
            SetToolstripButtonsEnabled(toolstrip, true);
        }

        private void UpdateButtons()
        {
            EnableButtons(this);
            EnableToolstripButtons(toolStrip1);


            if (!tree.AllNodes.Any())
                btnSave.Enabled = btnSaveAs.Enabled = btnRun.Enabled = false;

            if (tree.SelectedNode == null)
            {
                btnDetailsCurr.Enabled = false;
                btnNewTask.Enabled = btnDetailsTask.Enabled = false;
                return;
            }

            SchoolTreeNode selected = tree.SelectedNode.Tag as SchoolTreeNode;
            Debug.Assert(selected != null);

            if (selected is CurriculumNode)
                btnDetailsTask.Enabled = false;
        }

        #endregion UI

        #region DragDrop

        private void tree_ItemDrag(object sender, System.Windows.Forms.ItemDragEventArgs e)
        {
            tree.DoDragDropSelectedNodes(DragDropEffects.Move);
        }

        private void tree_DragOver(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(typeof(TreeNodeAdv[])) && tree.DropPosition.Node != null)
            {
                TreeNodeAdv draggedNode = (e.Data.GetData(typeof(TreeNodeAdv[])) as TreeNodeAdv[]).First();
                TreeNodeAdv parent = tree.DropPosition.Node;
                if (tree.DropPosition.Position != NodePosition.Inside)
                    parent = parent.Parent;

                if (IsNodeAncestor(draggedNode, parent))
                {
                    e.Effect = DragDropEffects.None;
                    return;
                }

                CurriculumNode curr = draggedNode.Tag as CurriculumNode;
                LearningTaskNode lt = draggedNode.Tag as LearningTaskNode;

                if (curr != null && parent.Level > 0) // curriculum can only be moved - not set as someone's child
                    e.Effect = DragDropEffects.None;
                else if (lt != null && parent.Level != 1)    //LT can only be in curriculum. Not in root, not in other LT
                    e.Effect = DragDropEffects.None;
                else
                    e.Effect = e.AllowedEffect;
            }
        }

        private bool IsNodeAncestor(TreeNodeAdv node, TreeNodeAdv examine)
        {
            while (examine != null)
            {
                if (node == examine)
                    return true;
                examine = examine.Parent;
            }
            return false;
        }

        private void tree_DragDrop(object sender, DragEventArgs e)
        {
            tree.BeginUpdate();

            TreeNodeAdv[] nodes = (TreeNodeAdv[])e.Data.GetData(typeof(TreeNodeAdv[]));
            if (tree.DropPosition.Node == null)
                return;
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
            UpdateWindowName(null, EventArgs.Empty);
        }

        #endregion DragDrop

        #region Button clicks

        private void btnNew_Click(object sender, EventArgs e)
        {
            Properties.School.Default.LastOpenedFile = null;
            Properties.School.Default.Save();
            m_currentFile = null;
            m_lastOpenedFile = null;
            m_savedRepresentation = null;
            m_model.Nodes.Clear();
            UpdateWindowName(null, EventArgs.Empty);
        }

        private void btnNewCurr_Click(object sender, EventArgs e)
        {
            CurriculumNode node = new CurriculumNode { Text = "Curr" + m_model.Nodes.Count.ToString() };
            m_model.Nodes.Add(node);
            UpdateButtons();    //for activating Run button - workaround because events of tree model are not working as expected

            // Curriculum name directly editable upon creation
            tree.SelectedNode = tree.FindNodeByTag(node);
            NodeTextBox control = (NodeTextBox)tree.NodeControls.ElementAt(1);
            control.BeginEdit();
        }

        private void btnDeleteCurr_Click(object sender, EventArgs e)
        {
            if (tree.SelectedNode.Tag is CurriculumNode)
            {
                DeleteNodes(sender, e);
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

            AddTaskView = new LearningTaskSelectionForm();
            AddTaskView.StartPosition = FormStartPosition.CenterParent;
            AddTaskView.ShowDialog(this);
            if (AddTaskView.ResultLearningTaskTypes == null)
                return;

            List<LearningTaskNode> newLearningTaskNodes = new List<LearningTaskNode>();
            foreach (Type learningTaskType in AddTaskView.ResultLearningTaskTypes)
            {
                newLearningTaskNodes.Add(new LearningTaskNode(learningTaskType, AddTaskView.ResultWorldType));
            }
            //LearningTaskNode task = new LearningTaskNode(AddTaskView.ResultTaskType, AddTaskView.ResultWorldType);

            if (newLearningTaskNodes.Count > 0)
            { 
                if (tree.SelectedNode.Tag is CurriculumNode)
                {
                    foreach (LearningTaskNode node in newLearningTaskNodes)
                    {
                        (tree.SelectedNode.Tag as Node).Nodes.Add(node);
                    }
                    tree.SelectedNode.IsExpanded = true;
                }
                else if (tree.SelectedNode.Tag is LearningTaskNode)
                {
                    LearningTaskNode source = tree.SelectedNode.Tag as LearningTaskNode;
                    int targetPosition = source.Parent.Nodes.IndexOf(source);
                    foreach (LearningTaskNode node in newLearningTaskNodes)
                    {
                        source.Parent.Nodes.Insert(++targetPosition, node);
                    }
                }
            }
        }

        private void DeleteNodes(object sender, EventArgs e)
        {
            // Walking through the nodes backwards. That way the index doesn't increase past the node size
            for (int i = tree.SelectedNodes.Count - 1; i >= 0; i--)
            {
                // After 1/many nodes are deleted, select the node that was after it/them
                if (i == tree.SelectedNodes.Count - 1)
                {
                    TreeNodeAdv nextone = tree.SelectedNode.NextNode;
                    if (nextone != null)
                    {
                        nextone.IsSelected = true;
                    }
                }

                TreeNodeAdv n = (TreeNodeAdv)tree.SelectedNodes[i];
                (n.Tag as Node).Parent = null;
            }
        }

        private void checkBoxAutosave_CheckedChanged(object sender, EventArgs e)
        {
            Properties.School.Default.AutosaveEnabled = (sender as CheckBox).Checked;
            Properties.School.Default.Save();
        }

        private void checkBoxAutorun_CheckedChanged(object sender, EventArgs e)
        {
            Properties.School.Default.AutorunEnabled = (sender as CheckBox).Checked;
            Properties.School.Default.Save();
        }

        private void btnRun_Click(object sender, EventArgs e)
        {
            //OpenFloatingOrActivate(RunView, DockPanel);
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

            if (ltNodes.Count() <= 0)
            {
                MessageBox.Show("The simulation cannot start because no active learning tasks were found, add at least one learning task", "Validation Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else
            { 
                OpenFloatingOrActivate(RunView, DockPanel);
         
                foreach (LearningTaskNode ltNode in ltNodes)
                    data.Add(ltNode);
                RunView.Data = data;
                RunView.Design = m_design;
                if (activeCurricula.Count() == 1)
                    RunView.RunName = activeCurricula.First().Text;
                else
                    RunView.RunName = Path.GetFileNameWithoutExtension(m_currentFile);
                RunView.Ready();
            }
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
            if (String.IsNullOrEmpty(m_currentFile))
                SaveProjectAs(sender, e);  // ask for file name and then save the project
            else
                SaveProject(m_currentFile);
        }

        private void btnImport_Click(object sender, EventArgs e)
        {
            AddFileContent();
        }

        private void btnDetailsTask_Click(object sender, EventArgs e)
        {
            if (tree.SelectedNode == null)
                return;

            LearningTaskNode node = tree.SelectedNode.Tag as LearningTaskNode;
            if (node == null)
                return;

            SchoolTaskDetailsForm detailsForm = new SchoolTaskDetailsForm(node.TaskType);
            OpenFloatingOrActivate(detailsForm, DockPanel);
        }

        private void btnDetailsCurr_Click(object sender, EventArgs e)
        {
            if (tree.SelectedNode == null)
                return;

            CurriculumNode curr = tree.SelectedNode.Tag as CurriculumNode;
            if (curr == null)
            {
                curr = tree.SelectedNode.Parent.Tag as CurriculumNode;
                if (curr == null)
                    return;
            }
            SchoolCurrDetailsForm detailsForm = new SchoolCurrDetailsForm(curr);
            OpenFloatingOrActivate(detailsForm, DockPanel);
        }

        private void btnToggleCheck(object sender, EventArgs e)
        {
            (sender as ToolStripButton).Checked = !(sender as ToolStripButton).Checked;
        }

        #endregion Button clicks

        #region (De)serialization

        private void SaveProject(string path)
        {
            string xmlResult = m_serializer.Serialize(m_design);
            File.WriteAllText(path, xmlResult);
            MyLog.Writer.WriteLine(MyLogLevel.INFO, "School project saved to: " + path);
            m_savedRepresentation = xmlResult;
            m_currentFile = path;
            UpdateWindowName(null, EventArgs.Empty);
        }

        private void SaveProjectAs(object sender, EventArgs e)
        {
            if (saveFileDialog1.ShowDialog() != DialogResult.OK)
                return;

            SaveProject(saveFileDialog1.FileName);
            Properties.School.Default.LastOpenedFile = saveFileDialog1.FileName;
            Properties.School.Default.Save();
            UpdateWindowName(null, EventArgs.Empty);
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
            m_savedRepresentation = xmlCurr;
            m_currentFile = filePath;
            UpdateWindowName(null, EventArgs.Empty);
            UpdateButtons();
            return false;
        }

        #endregion (De)serialization

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
                DeleteNodes(sender, null);
        }
    }
}
