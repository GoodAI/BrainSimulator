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
using System.Linq;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.School.GUI
{
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
                return (SchoolWorld)m_mainForm.Project.World;
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
            m_mainForm = mainForm;
            InitializeComponent();

            // here so it does not interfere with designer generated code
            btnRun.Click += m_mainForm.runToolButton_Click;
            btnStop.Click += m_mainForm.stopToolButton_Click;
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
            if (!Visible)
                return;
            if (!(m_mainForm.Project.World is SchoolWorld))
                m_mainForm.SelectWorldInWorldList(typeof(SchoolWorld));
            if (Design != null)
                PrepareSimulation(null, EventArgs.Empty);
        }

        private void SimulationHandler_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            if (!Visible)
                return;
            ILearningTask runningTask = m_school.CurrentLearningTask;
            if (runningTask != null && CurrentTask != null)
                UpdateTaskData(runningTask);
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
            if (Properties.School.Default.AutorunEnabled && Data != null)
                btnRun.PerformClick();
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
            else if (column == statusDataGridViewTextBoxColumn)
            {
                TrainingResult result = (TrainingResult)e.Value;
                DescriptionAttribute displayNameAtt = result.GetType().GetMember(result.ToString())[0].GetCustomAttributes(typeof(DescriptionAttribute), true).FirstOrDefault() as DescriptionAttribute;
                if (displayNameAtt != null)
                    e.Value = displayNameAtt.Description;
            }
        }

        private void SchoolRunForm_KeyDown(object sender, KeyEventArgs e)
        {
            switch (e.KeyCode)
            {
                case Keys.F5:
                    {
                        btnRun.PerformClick();
                        break;
                    }
                case Keys.F7:
                    {
                        btnPause.PerformClick();
                        break;
                    }
                case Keys.F8:
                    {
                        btnStop.PerformClick();
                        break;
                    }
                case Keys.F10:
                    {
                        btnStepOver.PerformClick();
                        break;
                    }
            }
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
            Properties.School.Default.ShowVisual = (sender as ToolStripButton).Checked = !(sender as ToolStripButton).Checked;
            Properties.School.Default.Save();

            splitContainer2.Panel2Collapsed = !Properties.School.Default.ShowVisual;
            SetObserver();
        }

        private void btnEmulateSuccess_Click(object sender, EventArgs e)
        {
            m_emulateSuccess = (sender as ToolStripButton).Checked = !(sender as ToolStripButton).Checked;
        }

        private void label4_Click(object sender, EventArgs e)
        {

        }

        private void SchoolRunForm_VisibleChanged(object sender, EventArgs e)
        {
            if (!Visible)
                return;
            SelectSchoolWorld(null, EventArgs.Empty);
            btnObserver.Checked = Properties.School.Default.ShowVisual;
            splitContainer2.Panel2Collapsed = !Properties.School.Default.ShowVisual;
            m_emulateSuccess = btnEmulateSuccess.Checked;
            UpdateWorldHandlers(null, m_mainForm.Project.World as SchoolWorld);
        }
    }
}
