using Aga.Controls.Tree;
using GoodAI.BrainSimulator.Forms;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
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
    public partial class SchoolRunForm : DockContent
    {
        public List<LearningTaskNode> Data;
        public List<LevelNode> Levels;
        public List<List<AttributeNode>> Attributes;
        public List<List<int>> AttributesChange;
        public PlanDesign Design;

        private const string DEFAULT_FORM_NAME = "School for AI";
        private readonly MainForm m_mainForm;
        private List<DataGridView> LevelGrids;
        private ObserverForm m_observer;

        private int m_currentRow = -1;
        private int m_stepOffset = 0;
        private Stopwatch m_currentLtStopwatch;

        private string m_autosaveFilePath;
        private YAXSerializer m_serializer;
        private TreeModel m_model;
        private string m_lastOpenedFile;
        private string m_uploadedRepresentation;
        private string m_savedRepresentation;
        private string m_currentFile;

        public event EventHandler WorkspaceChanged = delegate { };

        public LearningTaskSelectionForm AddTaskView { get; private set; }
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

        private LearningTaskNode CurrentTask
        {
            get
            {
                if (m_currentRow < 0 || m_currentRow >= Data.Count)
                    return null;
                return Data.ElementAt(m_currentRow);
            }
        }

        private IEnumerable<CurriculumNode> ActiveCurricula
        {
            get
            {
                return m_model.Nodes.Where(x => x is CurriculumNode).Select(x => x as CurriculumNode).Where(x => x.IsChecked == true);
            }
        }

        private string CurrentProjectName
        {
            get
            {
                return ActiveCurricula.Count() == 1 ? ActiveCurricula.First().Text : Path.GetFileNameWithoutExtension(m_currentFile);
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

                if (Data.Count > dataIndex)
                    return Data[dataIndex];

                return null;
            }
        }

        private PlanDesign m_design
        {
            get
            {
                return new PlanDesign(m_model.Nodes.Where(x => x is CurriculumNode).Select(x => x as CurriculumNode).ToList());
            }
        }

        //public SchoolRunForm RunView { get; private set; }

        private bool IsProjectUploaded
        {
            get
            {
                if (m_uploadedRepresentation == null)
                    return false;
                string currentRepresentation = m_serializer.Serialize(m_design);
                return m_uploadedRepresentation.Equals(currentRepresentation);
            }
        }

        private bool IsWorkspaceSaved
        {
            get
            {
                if (m_savedRepresentation == null)
                    return false;
                string currentRepresentation = m_serializer.Serialize(m_design);
                return m_savedRepresentation.Equals(currentRepresentation);
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

            if (!String.IsNullOrEmpty(Properties.School.Default.AutosaveFolder))
                btnAutosave.Checked = Properties.School.Default.AutosaveEnabled;
            else
                btnAutosave.Checked = false;

            m_lastOpenedFile = Properties.School.Default.LastOpenedFile;
            LoadCurriculum(m_lastOpenedFile);

            // school run form //

            // here so it does not interfere with designer generated code
            btnRun.Click += m_mainForm.runToolButton_Click;
            btnStop.Click += m_mainForm.stopToolButton_Click;
            btnPause.Click += m_mainForm.pauseToolButton_Click;
            btnStepOver.Click += m_mainForm.stepOverToolButton_Click;
            btnDebug.Click += m_mainForm.debugToolButton_Click;

            m_mainForm.SimulationHandler.StateChanged += SimulationHandler_StateChanged;
            m_mainForm.SimulationHandler.ProgressChanged += SimulationHandler_ProgressChanged;
            m_mainForm.WorldChanged += m_mainForm_WorldChanged;

            nodeTextBox1.DrawText += nodeTextBox1_DrawText;

            m_model.NodesChanged += ModelChanged;
            m_model.NodesInserted += ModelChanged;
            m_model.NodesRemoved += ModelChanged;

            WorkspaceChanged += SchoolRunForm_WorkspaceChanged;

            SchoolRunForm_WorkspaceChanged(null, EventArgs.Empty);
        }

        public void Ready()
        {
            UpdateGridData();
            PrepareSimulation(null, EventArgs.Empty);
            SetObserver();
        }

        public void UpdateGridData()
        {
            dataGridView1.DataSource = Data;
            dataGridView1.Invalidate();
        }

        private void SchoolRunForm_WorkspaceChanged(object sender, EventArgs e)
        {
            UpdateData();
            UpdateWindowName(null, EventArgs.Empty);
            UpdateButtons();
        }

        private void UpdateData()
        {
            if (!Visible || IsProjectUploaded)
                return;

            // update SchoolWorld
            SelectSchoolWorld(null, EventArgs.Empty);
            (m_mainForm.Project.World as SchoolWorld).Curriculum = m_design.AsSchoolCurriculum(m_mainForm.Project.World as SchoolWorld);

            // update curriculum detail grid
            List<LearningTaskNode> data = new List<LearningTaskNode>();
            IEnumerable<LearningTaskNode> ltNodes = ActiveCurricula.
                SelectMany(x => (x as CurriculumNode).Nodes).
                Select(x => x as LearningTaskNode).
                Where(x => x.IsChecked == true);

            foreach (LearningTaskNode ltNode in ltNodes)
                data.Add(ltNode);
            Data = data;
            Design = m_design;
            Ready();
        }

        // thanks to this, form will be emitter of event - not the underlying model (which could be confusing for subscribers)
        private void ModelChanged(object sender, EventArgs e)
        {
            WorkspaceChanged(this, EventArgs.Empty);
        }

        private string GetAutosaveFilename()
        {
            return CurrentProjectName + DateTime.Now.ToString("yyyy-MM-ddTHHmmss") + ".csv"; // ISO 8601
        }

        private void AddWorldHandlers(SchoolWorld world)
        {
            if (world == null)
                return;
            world.CurriculumStarting += PrepareSimulation;
            world.LearningTaskNew += GoToNextTask;
            world.LearningTaskNewLevel += UpdateLTLevel;
            world.LearningTaskFinished += LearningTaskFinished;
            world.TrainingUnitUpdated += UpdateTUStatus;
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
            world.TrainingUnitUpdated -= UpdateTUStatus;
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

            SetObserver();
        }

        private void UpdateButtons()
        {
            btnRun.Enabled = m_mainForm.runToolButton.Enabled;
            btnPause.Enabled = m_mainForm.pauseToolButton.Enabled;
            btnStop.Enabled = m_mainForm.stopToolButton.Enabled;

            EnableButtons(this);
            EnableToolstripButtons(toolStrip2);

            if (!tree.AllNodes.Any())
                btnSave.Enabled = btnSaveAs.Enabled = btnRun.Enabled = false;

            if (tree.SelectedNode == null)
            {
                btnNewTask.Enabled = btnDetails.Enabled = false;
                return;
            }

            Node selected = tree.SelectedNode.Tag as Node;
            Debug.Assert(selected != null);
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

                    m_observer.Observer.GenericTarget = m_school.Visual;
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

        #endregion UI

        private bool AddFileContent(bool clearWorkspace = false)
        {
            if (openFileDialog1.ShowDialog() != DialogResult.OK)
                return false;
            if (clearWorkspace)
                m_model.Nodes.Clear();
            LoadCurriculum(openFileDialog1.FileName);
            return true;
        }

        #region (De)serialization

        private void SaveProject(string path)
        {
            string xmlResult = m_serializer.Serialize(m_design);
            File.WriteAllText(path, xmlResult);
            MyLog.Writer.WriteLine(MyLogLevel.INFO, "School project saved to: " + path);
            m_savedRepresentation = xmlResult;
            m_currentFile = path;
            UpdateWindowName(null, EventArgs.Empty);
            UpdateData();
        }

        private void LoadCurriculum(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                return;

            string xmlCurr;
            try { xmlCurr = File.ReadAllText(filePath); }
            catch (IOException e)
            {
                MyLog.WARNING.WriteLine("Unable to read file " + filePath);
                return;
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
                return;
            }

            Properties.School.Default.LastOpenedFile = filePath;
            Properties.School.Default.Save();
            m_savedRepresentation = xmlCurr;
            m_currentFile = filePath;
        }

        #endregion (De)serialization

        private void disableLearningTaskPanel()
        {
            splitContainer3.Panel1.Enabled = false;
        }

        private void enableLearningTaskPanel()
        {
            splitContainer3.Panel1.Enabled = true;
        }

        private void ExportDataGridViewData(string filename, TextDataFormat format = TextDataFormat.CommaSeparatedValue)
        {
            IDataObject objectSave = Clipboard.GetDataObject();
            bool multiSelectAllowed = dataGridView1.MultiSelect;
            DataGridViewClipboardCopyMode copyMode = dataGridView1.ClipboardCopyMode;

            dataGridView1.ClipboardCopyMode = DataGridViewClipboardCopyMode.EnableAlwaysIncludeHeaderText;
            dataGridView1.MultiSelect = true;
            dataGridView1.SelectAll();
            Clipboard.SetDataObject(dataGridView1.GetClipboardContent());
            File.WriteAllText(filename, Clipboard.GetText(format));

            dataGridView1.MultiSelect = multiSelectAllowed;
            dataGridView1.ClipboardCopyMode = copyMode;
            if (objectSave != null)
                Clipboard.SetDataObject(objectSave);
        }
    }
}
