using Aga.Controls.Tree;
using Aga.Controls.Tree.NodeControls;
using GoodAI.BrainSimulator.Forms;
using GoodAI.Modules.School.Common;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.School.GUI
{
    [BrainSimUIExtension]
    public partial class SchoolMainForm : DockContent
    {
        public SchoolAddTaskForm AddTaskView { get; private set; }

        public class SchoolTreeNode : Node
        {
            public bool Checked { get; set; }

            public SchoolTreeNode(string text) : base(text) { }
        }

        public class CurriculumNode : SchoolTreeNode
        {
            public CurriculumNode(string text) : base(text) { }
        }

        public class LearningTaskNode : SchoolTreeNode
        {
            public Type Type { get; set; }
            public LearningTaskNode(string text) : base(text) { }
        }

        private TreeModel m_model;

        public SchoolMainForm()
        {
            AddTaskView = new SchoolAddTaskForm();

            InitializeComponent();

            m_model = new TreeModel();
            tree.Model = m_model;

            tree.BeginUpdate();
            for (int i = 0; i < 3; i++)
            {
                CurriculumNode node = AddCurriculum();
                for (int n = 0; n < 5; n++)
                {
                    LearningTaskNode child = AddLearningTask(node);
                }
            }
            tree.EndUpdate();

            tree.Refresh();
            UpdateButtons();
        }

        private void SchoolMainForm_Load(object sender, System.EventArgs e)
        {

        }

        private CurriculumNode AddCurriculum()
        {
            CurriculumNode node = new CurriculumNode("Curr" + m_model.Nodes.Count.ToString());
            m_model.Nodes.Add(node);
            return node;
        }

        private LearningTaskNode AddLearningTask(CurriculumNode parent)
        {
            LearningTaskNode node = new LearningTaskNode("LT " + parent.Nodes.Count.ToString());
            parent.Nodes.Add(node);
            return node;
        }

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
            btnNewCurr.Visible = true;
            if (tree.SelectedNode == null)
                return;

            SchoolTreeNode selected = tree.SelectedNode.Tag as SchoolTreeNode;
            Debug.Assert(selected != null);

            if (selected is CurriculumNode)
                groupBox1.Text = "Curriculum";
            else if (selected is LearningTaskNode)
                groupBox1.Text = "Learning task";

            btnDelete.Visible = true;
            btnNewCurr.Visible = btnDetailsCurr.Visible = btnExportCurr.Visible = btnImportCurr.Visible = btnNewTask.Visible = btnRun.Visible =
                selected is CurriculumNode;
            btnDetailsTask.Visible = selected is LearningTaskNode;
        }

        private void btnNewCurr_Click(object sender, EventArgs e)
        {
            CurriculumNode newCurr = AddCurriculum();
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

        private void btnNewTask_Click(object sender, EventArgs e)
        {
            if (tree.SelectedNode == null || !(tree.SelectedNode.Tag is CurriculumNode))
                return;

            AddTaskView.ShowDialog(this);
            if (AddTaskView.ResultTask == null)
                return;

            LearningTaskNode newTask = new LearningTaskNode(AddTaskView.ResultTask);
            newTask.Type = AddTaskView.ResultTaskType;
            (tree.SelectedNode.Tag as Node).Nodes.Add(newTask);
            tree.SelectedNode.IsExpanded = true;
        }

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

        private void DeleteNode()
        {
            if (tree.SelectedNode != null && tree.SelectedNode.Tag is Node)
                (tree.SelectedNode.Tag as Node).Parent = null;
        }

        private void btnDelete_Click(object sender, EventArgs e)
        {
            DeleteNode();
        }
    }
}
