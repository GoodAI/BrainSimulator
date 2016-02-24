using Aga.Controls.Tree;
using Aga.Controls.Tree.NodeControls;
using GoodAI.BrainSimulator.Forms;
using GoodAI.Core.Execution;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Collections.Generic;
using System.ComponentModel;
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
    public partial class SchoolRunForm : DockContent
    {
        public List<LearningTaskNode> Data;
        public List<LevelNode> Levels;
        public List<List<AttributeNode>> Attributes;
        public List<List<int>> AttributesChange;
        public PlanDesign Design;

        private List<DataGridView> LevelGrids;

        private readonly MainForm m_mainForm;
        private string m_runName;
        private ObserverForm m_observer;

        private int m_currentRow = -1;
        private int m_stepOffset = 0;
        private Stopwatch m_currentLtStopwatch;

        private int m_numberOfTU;

        private bool m_showObserver { get { return btnObserver.Checked; } }
        private bool m_emulateSuccess
        {
            set
            {
                if (m_school != null)
                    m_school.EmulatedUnitSuccessProbability = value ? 1f : 0f;
            }
        }

        private SchoolWorld m_school
        {
            get
            {
                return m_mainForm.Project.World as SchoolWorld;
            }
        }

        public string RunName
        {
            get { return m_runName; }
            set
            {
                m_runName = value;

                Text = String.IsNullOrEmpty(m_runName) ? "School run" : "School run - " + m_runName;
            }
        }

        private LearningTaskNode CurrentTask
        {
            get
            {
                if (m_currentRow < 0 || m_currentRow >= Data.Count)
                    return null;
                return Data.ElementAt(m_currentRow);
            }
        }


        public SchoolRunForm(MainForm mainForm)
        {
            // school main form //

            m_serializer = new YAXSerializer(typeof(PlanDesign));
            m_mainForm = mainForm;
            //RunView = new SchoolRunForm(m_mainForm);

            InitializeComponent();

            m_model = new TreeModel();
            tree.Model = m_model;
            tree.Refresh();

            btnAutosave.Checked = Properties.School.Default.AutosaveEnabled;
            m_lastOpenedFile = Properties.School.Default.LastOpenedFile;
            if (LoadCurriculum(m_lastOpenedFile))
                saveFileDialog1.FileName = m_lastOpenedFile;

            UpdateButtonsSR();


            // school run form //

            m_mainForm = mainForm;

            // here so it does not interfere with designer generated code
            btnPause.Click += m_mainForm.pauseToolButton_Click;
            btnStepOver.Click += m_mainForm.stepOverToolButton_Click;
            btnDebug.Click += m_mainForm.debugToolButton_Click;

            m_mainForm.SimulationHandler.StateChanged += SimulationHandler_StateChanged;
            m_mainForm.SimulationHandler.ProgressChanged += SimulationHandler_ProgressChanged;
            m_mainForm.WorldChanged += m_mainForm_WorldChanged;
            m_mainForm.WorldChanged += SelectSchoolWorld;

            UpdateButtons();
        }

        void SimulationHandler_StateChanged(object sender, MySimulationHandler.StateEventArgs e)
        {
            if (m_currentLtStopwatch != null)
                if (e.NewState == MySimulationHandler.SimulationState.PAUSED)
                    m_currentLtStopwatch.Stop();
                else if (e.NewState == MySimulationHandler.SimulationState.RUNNING ||
                    e.NewState == MySimulationHandler.SimulationState.RUNNING_STEP)
                    m_currentLtStopwatch.Start();

            UpdateButtons();
        }

        void m_mainForm_WorldChanged(object sender, MainForm.WorldChangedEventArgs e)
        {
            UpdateWorldHandlers(e.OldWorld as SchoolWorld, e.NewWorld as SchoolWorld);
        }

        private void SelectSchoolWorld(object sender, EventArgs e)
        {

        }

        private void SimulationHandler_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            if (!Visible)
                return;

            Invoke((MethodInvoker)(() =>
            {
                ILearningTask runningTask = m_school.CurrentLearningTask;
                if (runningTask != null && CurrentTask != null)
                    UpdateTaskData(runningTask);
            }));
        }

        private void AddWorldHandlers(SchoolWorld world)
        {
            if (world == null)
                return;
            world.CurriculumStarting += PrepareSimulation;
            world.LearningTaskNew += GoToNextTask;
            world.LearningTaskNewLevel += UpdateLTLevel;
            world.LearningTaskFinished += LearningTaskFinished;
            world.TrainingUnitFinished += UpdateTUStatus;
            world.TrainingUnitFinished += UpdateTrainingUnitNumber;
        }

        private void RemoveWorldHandlers(SchoolWorld world)
        {
            if (world == null)
                return;
            world.CurriculumStarting -= PrepareSimulation;
            world.LearningTaskNew -= GoToNextTask;
            world.LearningTaskNewLevel -= UpdateLTLevel;
            world.LearningTaskFinished -= LearningTaskFinished;
            world.TrainingUnitFinished -= UpdateTUStatus;
            world.TrainingUnitFinished -= UpdateTrainingUnitNumber;
        }

        private void UpdateWorldHandlers(SchoolWorld oldWorld, SchoolWorld newWorld)
        {
            if (!Visible)
                return;
            if (oldWorld != null)
                RemoveWorldHandlers(oldWorld as SchoolWorld);
            if (newWorld != null)
                AddWorldHandlers(newWorld as SchoolWorld);
        }

        private void LearningTaskFinished(object sender, SchoolEventArgs e)
        {
            m_numberOfTU = 0;
            UpdateTaskData(e.Task);
        }

        private void UpdateTrainingUnitNumber(object sender, SchoolEventArgs e)
        {
            Invoke((MethodInvoker)(() =>
            {
                unitNumberLabel.Text = m_numberOfTU++.ToString();
            }
            ));
        }

        private void UpdateLTLevel(object sender, SchoolEventArgs e)
        {
            if (tabControl1 != null && tabControl1.TabCount > 0)
            {
                Invoke((MethodInvoker)(() =>
                {
                    tabControl1.SelectedIndex = m_school.Level - 1;
                    currentLevelLabel.Text = m_school.Level.ToString();
                }
                ));
            }
        }

        private void UpdateTUStatus(object sender, EventArgs e)
        {
            Invoke((MethodInvoker)(() =>
            {
                actualRewardLabel.Text = m_school.Reward.ToString("F");
            }
            ));
        }

        private void UpdateButtons()
        {
            btnRun.Enabled = m_mainForm.runToolButton.Enabled;
            btnPause.Enabled = m_mainForm.pauseToolButton.Enabled;
            btnStop.Enabled = m_mainForm.stopToolButton.Enabled;
        }

        public void Ready()
        {
            UpdateGridData();
            PrepareSimulation(null, EventArgs.Empty);
            SetObserver();
            /*if (Properties.School.Default.AutorunEnabled && Data != null)
                btnRun.PerformClick();*/
        }

        public void UpdateGridData()
        {
            dataGridView1.DataSource = Data;
            dataGridView1.Invalidate();
        }

        private void UpdateTaskData(ILearningTask runningTask)
        {
            if (CurrentTask == null)
                return;
            CurrentTask.Steps = (int)m_mainForm.SimulationHandler.SimulationStep - m_stepOffset;
            CurrentTask.Progress = (int)runningTask.Progress;
            TimeSpan diff = m_currentLtStopwatch.Elapsed;
            CurrentTask.Time = (float)Math.Round(diff.TotalSeconds, 2);
            CurrentTask.Status = m_school.TaskResult;

            UpdateGridData();
        }

        private void GoToNextTask(object sender, SchoolEventArgs e)
        {
            m_currentRow++;
            m_stepOffset = (int)m_mainForm.SimulationHandler.SimulationStep;
            m_currentLtStopwatch = new Stopwatch();
            m_currentLtStopwatch.Start();

            HighlightCurrentTask();
        }

        private void SetObserver()
        {
            if (m_showObserver)
            {
                if (m_observer == null)
                {
                    try
                    {
                        MyMemoryBlockObserver observer = new MyMemoryBlockObserver();
                        observer.Target = m_school.Visual;

                        if (observer == null)
                            throw new InvalidOperationException("No observer was initialized");

                        m_observer = new ObserverForm(m_mainForm, observer, m_school);

                        m_observer.TopLevel = false;
                        observerDockPanel.Controls.Add(m_observer);

                        m_observer.CloseButtonVisible = false;
                        m_observer.MaximizeBox = false;
                        m_observer.Size = observerDockPanel.Size + new System.Drawing.Size(16, 38);
                        m_observer.Location = new System.Drawing.Point(-8, -30);

                        m_observer.Show();
                    }
                    catch (Exception e)
                    {
                        MyLog.ERROR.WriteLine("Error creating observer: " + e.Message);
                    }
                }
                else
                {
                    m_observer.Show();
                    observerDockPanel.Show();
                }
            }
            else
            {
                if (m_observer != null)
                {
                    observerDockPanel.Hide();
                }
            }
        }

        private void HighlightCurrentTask()
        {
            DataGridViewCellStyle defaultStyle = new DataGridViewCellStyle();
            DataGridViewCellStyle highlightStyle = new DataGridViewCellStyle();
            highlightStyle.BackColor = Color.PaleGreen;

            dataGridView1.Rows[m_currentRow].Selected = true;
            foreach (DataGridViewRow row in dataGridView1.Rows)
                foreach (DataGridViewCell cell in row.Cells)
                    if (row.Index == m_currentRow)
                        cell.Style = highlightStyle;
                    else
                        cell.Style = defaultStyle;
        }

        private void PrepareSimulation(object sender, EventArgs e)
        {
            // data
            m_school.Curriculum = Design.AsSchoolCurriculum(m_school);

            // gui
            m_stepOffset = 0;
            m_currentRow = -1;
            Data.ForEach(x => { x.Steps = x.Progress = 0; x.Time = 0f; x.Status = TrainingResult.None; });

            string xmlResult = m_serializer.Serialize(m_design);
            m_uploadedRepresentation = xmlResult;
        }

        private void dataGridView1_CellFormatting(object sender, DataGridViewCellFormattingEventArgs e)
        {
            DataGridViewColumn column = dataGridView1.Columns[e.ColumnIndex];

            if ((column == TaskType || column == WorldType) && e.Value != null)
            {
                // I am not sure about how bad this approach is, but it get things done
                Type typeValue = e.Value as Type;

                DisplayNameAttribute displayNameAtt = typeValue.GetCustomAttributes(typeof(DisplayNameAttribute), true).FirstOrDefault() as DisplayNameAttribute;
                if (displayNameAtt != null)
                    e.Value = displayNameAtt.DisplayName;
                else
                    e.Value = typeValue.Name;
            }
            else if (column == statusDataGridViewTextBoxColumn1)
            {
                TrainingResult result = (TrainingResult)e.Value;
                DescriptionAttribute displayNameAtt = result.GetType().GetMember(result.ToString())[0].GetCustomAttributes(typeof(DescriptionAttribute), true).FirstOrDefault() as DescriptionAttribute;
                if (displayNameAtt != null)
                    e.Value = displayNameAtt.Description;
            }
        }

        private void SchoolRunForm_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Handled)
                return;

            switch (e.KeyCode)
            {
                case Keys.F5:
                    {
                        if (!m_mainForm.runToolButton.Enabled)
                            return;
                        btnRun.PerformClick();
                        break;
                    }
                case Keys.F7:
                    {
                        if (!m_mainForm.pauseToolButton.Enabled)
                            return;
                        btnPause.PerformClick();
                        break;
                    }
                case Keys.F8:
                    {
                        if (!m_mainForm.stopToolButton.Enabled)
                            return;
                        btnStop.PerformClick();
                        break;
                    }
                case Keys.F10:
                    {
                        if (!m_mainForm.stepOverToolButton.Enabled)
                            return;
                        btnStepOver.PerformClick();
                        break;
                    }
                case Keys.Delete:
                    {
                        DeleteNodes(sender, null);
                        break;
                    }
                default:
                    {
                        return;
                    }
            }
            e.Handled = true;
        }

        private LearningTaskNode SelectedLearningTask
        {
            get
            {
                int dataIndex;
                if (dataGridView1.SelectedRows != null && dataGridView1.SelectedRows.Count > 0)
                {
                    DataGridViewRow row = dataGridView1.SelectedRows[0];
                    dataIndex = row.Index;
                }
                else
                {
                    dataIndex = 0;
                }
                return Data[dataIndex];
            }
        }

        private void dataGridView1_SelectionChanged(object sender, EventArgs e)
        {
            Invoke((MethodInvoker)(() =>
                {
                    LearningTaskNode ltNode = SelectedLearningTask;
                    Type ltType = ltNode.TaskType;
                    ILearningTask lt = LearningTaskFactory.CreateLearningTask(ltType);
                    TrainingSetHints hints = lt.TSProgression[0];

                    Levels = new List<LevelNode>();
                    LevelGrids = new List<DataGridView>();
                    Attributes = new List<List<AttributeNode>>();
                    AttributesChange = new List<List<int>>();
                    tabControl1.TabPages.Clear();

                    for (int i = 0; i < lt.TSProgression.Count; i++)
                    {
                        // create tab
                        LevelNode ln = new LevelNode(i + 1);
                        Levels.Add(ln);
                        TabPage tp = new TabPage(ln.Text);
                        tabControl1.TabPages.Add(tp);

                        // create grid
                        DataGridView dgv = new DataGridView();

                        dgv.Parent = tp;
                        dgv.Margin = new Padding(3);
                        dgv.Dock = DockStyle.Fill;
                        dgv.RowHeadersVisible = false;
                        dgv.SelectionMode = DataGridViewSelectionMode.FullRowSelect;
                        dgv.AllowUserToResizeRows = false;
                        // create attributes
                        Attributes.Add(new List<AttributeNode>());
                        if (i > 0)
                        {
                            hints.Set(lt.TSProgression[i]);
                        }
                        foreach (var attribute in hints)
                        {
                            AttributeNode an = new AttributeNode(attribute.Key, attribute.Value);
                            Attributes[i].Add(an);
                            // create tooltips
                        }

                        Attributes[i].Sort(Comparer<AttributeNode>.Create((x, y) => x.Name.CompareTo(y.Name)));
                        dgv.DataSource = Attributes[i];


                        dgv.Columns[0].Width = 249;
                        dgv.Columns[0].ReadOnly = true;
                        dgv.Columns[1].ReadOnly = true;

                        AttributesChange.Add(new List<int>());
                        if (i > 0)
                        {
                            foreach (var attribute in lt.TSProgression[i])
                            {
                                int attributeIdx = Attributes[i].IndexOf(new AttributeNode(attribute.Key.Name));
                                AttributesChange[i].Add(attributeIdx);
                            }
                        }

                        LevelGrids.Add(dgv);
                        dgv.ColumnWidthChanged += levelGridColumnSizeChanged;
                        dgv.CellFormatting += lGrid_CellFormatting;
                        dgv.SelectionChanged += levelGridSelectionChanged;

                        tabControl1.Update();
                    }
                }
            ));
        }

        private void levelGridSelectionChanged(object sender, EventArgs e)
        {
            DataGridView dgv = sender as DataGridView;
            dgv.ClearSelection();
        }

        private void lGrid_CellFormatting(object sender, DataGridViewCellFormattingEventArgs args)
        {
            // colouring changes between levels
            DataGridView dgv = sender as DataGridView;
            int level = LevelGrids.IndexOf(dgv);
            if (AttributesChange.Count == 0)
            {
                return;
            }
            if (AttributesChange[level].Contains(args.RowIndex))
            {
                args.CellStyle.BackColor = Color.LightGreen;
            }

            // set tooltip text
            int row = args.RowIndex;
            int column = args.ColumnIndex;
            dgv.Rows[row].Cells[column].ToolTipText = Attributes[level][row].GetAnotation();

            // unselect dgv
            dgv.ClearSelection();
        }

        private void levelGridColumnSizeChanged(object sender, DataGridViewColumnEventArgs e)
        {
            DataGridView dg = sender as DataGridView;
            for (int i = 0; i < dg.Columns.Count; i++)
            {
                int width = dg.Columns[i].Width;
                foreach (var levelGrid in LevelGrids)
                {
                    if (dg == levelGrid) continue;
                    levelGrid.Columns[i].Width = width;
                }
            }
        }

        private void btnObserver_Click(object sender, EventArgs e)
        {
            Properties.School.Default.ShowVisual = (sender as ToolStripButton).Checked;
            Properties.School.Default.Save();

            Invoke((MethodInvoker)(() =>
            {
                splitContainer2.Panel2Collapsed = !Properties.School.Default.ShowVisual;
                SetObserver();
                Invalidate();
            }));
        }

        private void btnEmulateSuccess_Click(object sender, EventArgs e)
        {
            m_emulateSuccess = (sender as ToolStripButton).Checked = !(sender as ToolStripButton).Checked;
        }

        private void SchoolRunForm_VisibleChanged(object sender, EventArgs e)
        {
            if (!Visible)
                return;
            SelectSchoolWorld(null, EventArgs.Empty);
            btnObserver.Checked = Properties.School.Default.ShowVisual;
            splitContainer2.Panel2Collapsed = !Properties.School.Default.ShowVisual;
            m_emulateSuccess = btnEmulateSuccess.Checked;
            SchoolWorld school = m_mainForm.Project.World as SchoolWorld;
            UpdateWorldHandlers(school, school);

            UpdateWindowName(sender, e);
            UpdateUploadState(sender, e);
        }

        private void showRunPanelStripButton_Click(object sender, EventArgs e)
        {
            splitContainer3.Panel2Collapsed = (sender as ToolStripButton).Checked;
        }


        // /////////////////////////////////////////////////////////////////////////////////// //
        // /////////////////////////////////////////////////////////////////////////////////// //
        // /////////////////////////////////////////////////////////////////////////////////// //
        // ///////////////////////////////// SCHOOL MAIN FORM //////////////////////////////// //
        // /////////////////////////////////////////////////////////////////////////////////// //
        // /////////////////////////////////////////////////////////////////////////////////// //
        // /////////////////////////////////////////////////////////////////////////////////// //


        private const string DEFAULT_FORM_NAME = "School for AI";

        private YAXSerializer m_serializer;
        private TreeModel m_model;
        private string m_lastOpenedFile;
        private string m_uploadedRepresentation;
        private string m_savedRepresentation;
        private string m_currentFile;

        private PlanDesign m_design
        {
            get
            {
                return new PlanDesign(m_model.Nodes.Where(x => x is CurriculumNode).Select(x => x as CurriculumNode).ToList());
            }
        }

        public LearningTaskSelectionForm AddTaskView { get; private set; }
        //public SchoolRunForm RunView { get; private set; }

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


        private void UpdateUploadState(object sender, EventArgs e)
        {
            if (!Visible)
            {
                return;
            }

            if (!IsProjectUploaded())
            {
                uploadLearningTasks();
            }

        }

        private bool IsWorkspaceSaved()
        {
            if (m_savedRepresentation == null)
                return false;
            string currentRepresentation = m_serializer.Serialize(m_design);
            return m_savedRepresentation.Equals(currentRepresentation);
        }

        private bool IsProjectUploaded()
        {
            if (m_uploadedRepresentation == null)
                return false;
            string currentRepresentation = m_serializer.Serialize(m_design);
            return m_uploadedRepresentation.Equals(currentRepresentation);
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

        private void UpdateButtonsSR()
        {
            EnableButtons(this);
            EnableToolstripButtons(toolStrip2);


            if (!tree.AllNodes.Any())
                btnSave.Enabled = btnSaveAs.Enabled = btnRun.Enabled = false;

            if (tree.SelectedNode == null)
            {
                btnNewTask.Enabled = btnDetails.Enabled = false;
                return;
            }

            SchoolTreeNode selected = tree.SelectedNode.Tag as SchoolTreeNode;
            Debug.Assert(selected != null);

            UpdateWindowName(null, EventArgs.Empty);
            UpdateUploadState(null, EventArgs.Empty);
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
            UpdateUploadState(null, EventArgs.Empty);
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
            m_uploadedRepresentation = null;
            m_model.Nodes.Clear();
            UpdateWindowName(null, EventArgs.Empty);
            UpdateUploadState(null, EventArgs.Empty);
        }

        private void btnNewCurr_Click(object sender, EventArgs e)
        {
            CurriculumNode node = new CurriculumNode { Text = "Curr" + m_model.Nodes.Count.ToString() };
            m_model.Nodes.Add(node);
            UpdateButtonsSR();    //for activating Run button - workaround because events of tree model are not working as expected

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

            UpdateWindowName(null, EventArgs.Empty);
            UpdateUploadState(null, EventArgs.Empty);
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

        private void btnAutosave_CheckedChanged(object sender, EventArgs e)
        {
            Properties.School.Default.AutosaveEnabled = (sender as ToolStripButton).Checked;
            Properties.School.Default.Save();
        }

        private void uploadLearningTasks()
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

            foreach (LearningTaskNode ltNode in ltNodes)
                data.Add(ltNode);
            Data = data;
            Design = m_design;
            /*if (activeCurricula.Count() == 1)
                RunName = activeCurricula.First().Text;
            else
                RunName = Path.GetFileNameWithoutExtension(m_currentFile);*/
            Ready();
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

        private void btnDetails_Click(object sender, EventArgs e)
        {
            if (tree.SelectedNode == null)
                return;

            DockContent detailsForm = null;
            if (tree.SelectedNode.Tag is CurriculumNode)
            {
                CurriculumNode curr = tree.SelectedNode.Tag as CurriculumNode;
                detailsForm = new SchoolCurrDetailsForm(curr);
            }
            else if (tree.SelectedNode.Tag is LearningTaskNode)
            {
                LearningTaskNode node = tree.SelectedNode.Tag as LearningTaskNode;
                detailsForm = new SchoolTaskDetailsForm(node.TaskType);
            }
            if (detailsForm != null)
                OpenFloatingOrActivate(detailsForm, DockPanel);
        }

        private void btnToggleCheck(object sender, EventArgs e)
        {
            (sender as ToolStripButton).Checked = !(sender as ToolStripButton).Checked;
        }

        private void btnUpload_Click(object sender, EventArgs e)
        {
            if (!(m_mainForm.Project.World is SchoolWorld))
                m_mainForm.SelectWorldInWorldList(typeof(SchoolWorld));
            (m_mainForm.Project.World as SchoolWorld).Curriculum = m_design.AsSchoolCurriculum(m_mainForm.Project.World as SchoolWorld);
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
            UpdateUploadState(null, EventArgs.Empty);
        }

        private void SaveProjectAs(object sender, EventArgs e)
        {
            if (saveFileDialog1.ShowDialog() != DialogResult.OK)
                return;

            SaveProject(saveFileDialog1.FileName);
            Properties.School.Default.LastOpenedFile = saveFileDialog1.FileName;
            Properties.School.Default.Save();
            UpdateWindowName(null, EventArgs.Empty);
            UpdateUploadState(null, EventArgs.Empty);
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
            UpdateUploadState(null, EventArgs.Empty);
            UpdateButtonsSR();
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
            UpdateButtonsSR();
        }

        private void nodeTextBox1_DrawText(object sender, DrawTextEventArgs e)
        {
            if (e.Node.IsSelected)
                e.Font = new System.Drawing.Font(e.Font, FontStyle.Bold);
        }

        private void tree_Click(object sender, EventArgs e)
        {
            uploadLearningTasks();
        }

        private void btnRun_Click(object sender, EventArgs e)
        {
            m_mainForm.runToolButton_Click(sender, e);
            if (m_mainForm.SimulationHandler.State == MySimulationHandler.SimulationState.RUNNING ||
                m_mainForm.SimulationHandler.State == MySimulationHandler.SimulationState.RUNNING_STEP ||
                m_mainForm.SimulationHandler.State == MySimulationHandler.SimulationState.PAUSED)
            {
                disableLearningTaskPanel();
            }
        }


        private void disableLearningTaskPanel()
        {
            splitContainer3.Panel1.Enabled = false;
        }

        private void btnStop_Click(object sender, EventArgs e)
        {
            m_mainForm.stopToolButton_Click(sender, e);
            enableLearningTaskPanel();
        }

        private void enableLearningTaskPanel()
        {
            splitContainer3.Panel1.Enabled = true;
        }

        /*
        private void SchoolMainForm_KeyDown(object sender, KeyEventArgs e)
        {
            switch (e.KeyCode)
            {
                case Keys.Delete:
                    {
                        DeleteNodes(sender, null);
                        break;
                    }
                case Keys.F5:
                    {
                        btnRun.PerformClick();
                        break;
                    }
            }
        }*/
    }
}
