using Aga.Controls.Tree;
using Aga.Controls.Tree.NodeControls;
using GoodAI.BrainSimulator.Forms;
using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
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
using System.Runtime.InteropServices;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.School.GUI
{
    [BrainSimUIExtension]
    public partial class SchoolRunForm : DockContent
    {
        private void SimulationHandler_StateChanged(object sender, MySimulationHandler.StateEventArgs e)
        {
            if (!Visible && !(m_mainForm.Project.World is SchoolWorld))
                return;
            // time measurements
            if (m_currentLtStopwatch != null)
                if (e.NewState == MySimulationHandler.SimulationState.PAUSED)
                    m_currentLtStopwatch.Stop();
                else if (e.NewState == MySimulationHandler.SimulationState.RUNNING ||
                    e.NewState == MySimulationHandler.SimulationState.RUNNING_STEP)
                    m_currentLtStopwatch.Start();

            // workspace panel enable/disable
            if (e.NewState == MySimulationHandler.SimulationState.RUNNING ||
                    e.NewState == MySimulationHandler.SimulationState.RUNNING_STEP ||
                    e.NewState == MySimulationHandler.SimulationState.PAUSED)
            {
                disableLearningTaskPanel();
            }
            else
            {
                enableLearningTaskPanel();
            }

            // autosave
            if (e.NewState == MySimulationHandler.SimulationState.PAUSED ||
                e.NewState == MySimulationHandler.SimulationState.STOPPED)
                if (Properties.School.Default.AutosaveEnabled)
                {
                    if (String.IsNullOrEmpty(m_autosaveFilePath))
                    {
                        string filename = GetAutosaveFilename();
                        m_autosaveFilePath = Path.Combine(Properties.School.Default.AutosaveFolder, filename);
                    }
                    ExportDataGridViewData(m_autosaveFilePath);
                }

            if (e.NewState == MySimulationHandler.SimulationState.STOPPED)
                m_autosaveFilePath = null;

            // buttons
            UpdateButtons();
        }

        private void m_mainForm_WorldChanged(object sender, MainForm.WorldChangedEventArgs e)
        {
            m_uploadedRepresentation = null;
            UpdateWorldHandlers(e.OldWorld as SchoolWorld, e.NewWorld as SchoolWorld);
            SelectSchoolWorld(null, EventArgs.Empty);
            UpdateData();
        }

        private void SelectSchoolWorld(object sender, EventArgs e)
        {
            if (!Visible || (m_mainForm.Project.World is SchoolWorld))
                return;
            m_mainForm.SelectWorldInWorldList(typeof(SchoolWorld));
            if (Design != null)
                PrepareSimulation(null, EventArgs.Empty);
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

        private void LearningTaskFinished(object sender, SchoolEventArgs e)
        {
            Invoke((MethodInvoker)(() =>
            {
                UpdateTaskData(e.Task);
            }));
        }

        private void VisualFormatChanged(object sender, SchoolEventArgs e)
        {
            Invoke((MethodInvoker)(() =>
            {
                if (m_observer != null)
                {
                    MyMemoryBlockObserver mbObserver = (m_observer.Observer as MyMemoryBlockObserver);
                    switch (m_school.Format)
                    {
                        case SchoolWorld.VisualFormat.Raw:
                            mbObserver.Method = RenderingMethod.Raw;
                            break;
                        case SchoolWorld.VisualFormat.RGB:
                            mbObserver.Method = RenderingMethod.RGB;
                            break;
                    }
                }
            }));
        }

        private void UpdateTrainingUnitNumber(object sender, SchoolEventArgs e)
        {
            Invoke((MethodInvoker)(() =>
            {
                labelTrainingUnitValue.Text = e.Task.CurrentNumberOfAttempts.ToString();
                labelSuccessefulAttemptsValue.Text = e.Task.CurrentNumberOfSuccesses.ToString() + " / " + e.Task.NumberOfSuccessesRequired;
            }
            ));
        }

        private void UpdateLTLevel(object sender, SchoolEventArgs e)
        {
            Invoke((MethodInvoker)(() =>
            {
                if (tabControlLevels != null && tabControlLevels.TabCount > 0)
                {
                    var focus = GetFocusedControl();
                    if (m_currentRow >= 0)
                    {
                        dataGridViewLearningTasks.Rows[m_currentRow].Selected = true;
                    }
                    tabControlLevels.SelectedIndex = m_school.Level - 1;
                    labelCurrentLevelValue.Text = m_school.Level.ToString() + " / " + e.Task.NumberOfLevels;

                    if (focus != null)
                    {
                        focus.Focus();
                    }
                    (tabControlLevels.SelectedTab.Controls[0] as DataGridView).ClearSelection();
                }
            }));
        }

        [DllImport("user32.dll", CharSet = CharSet.Auto, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr GetFocus();

        private Control GetFocusedControl()
        {
            Control focusedControl = null;
            // To get hold of the focused control:
            IntPtr focusedHandle = GetFocus();
            if (focusedHandle != IntPtr.Zero)
                // Note that if the focused Control is not a .Net control, then this will return null.
                focusedControl = Control.FromHandle(focusedHandle);
            return focusedControl;
        }

        private void UpdateTUStatus(object sender, EventArgs e)
        {
            Invoke((MethodInvoker)(() =>
            {
                m_school.RewardMB.SafeCopyToHost();
                labelPreviousStepRewardValue.Text = m_school.Reward.ToString("F");
            }
            ));
        }

        private void GoToNextTask(object sender, SchoolEventArgs e)
        {
            Invoke((MethodInvoker)(() =>
                {
                    m_currentRow++;
                    m_stepOffset = (int)m_mainForm.SimulationHandler.SimulationStep;
                    m_currentLtStopwatch = new Stopwatch();
                    m_currentLtStopwatch.Start();

                    HighlightCurrentTask();
                }));
        }

        private void PrepareSimulation(object sender, EventArgs e)
        {
            Invoke((MethodInvoker)(() =>
                {
                    // data
                    m_school.Curriculum = Design.AsSchoolCurriculum(m_school);

                    // gui
                    m_stepOffset = 0;
                    m_currentRow = -1;
                    Data.ForEach(x => { x.Steps = x.Progress = 0; x.Time = 0f; x.Status = TrainingResult.None; });

                    string xmlResult = m_serializer.Serialize(m_design);
                    m_uploadedRepresentation = xmlResult;
                }));
        }

        private void dataGridView1_CellFormatting(object sender, DataGridViewCellFormattingEventArgs e)
        {
            DataGridViewColumn column = dataGridViewLearningTasks.Columns[e.ColumnIndex];

            if ((column == TaskType || column == WorldType) && e.Value != null)
            {
                // I am not sure about how bad this approach is, but it get things done
                Type typeValue = e.Value as Type;

                DisplayNameAttribute displayNameAtt = typeValue.
                    GetCustomAttributes(typeof(DisplayNameAttribute), true).
                    FirstOrDefault() as DisplayNameAttribute;
                if (displayNameAtt != null)
                    e.Value = displayNameAtt.DisplayName;
                else
                    e.Value = typeValue.Name;
            }
            else if (column == statusDataGridViewTextBoxColumn)
            {
                TrainingResult result = (TrainingResult)e.Value;
                DescriptionAttribute displayNameAtt = result.GetType().
                    GetMember(result.ToString())[0].
                    GetCustomAttributes(typeof(DescriptionAttribute), true).
                    FirstOrDefault() as DescriptionAttribute;
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
                        btnDelete.PerformClick();
                        break;
                    }
                default:
                    {
                        return;
                    }
            }
            e.Handled = true;
        }

        private LearningTaskNode prevGridViewSelection;

        private void dataGridView1_SelectionChanged(object sender, EventArgs e)
        {
            Invoke((MethodInvoker)(() =>
            {

                LearningTaskNode ltNode = SelectedLearningTask;

                // if no selection, clear table and return
                if (ltNode == null)
                {
                    tabControlLevels.TabPages.Clear();
                    prevGridViewSelection = null;
                    return;
                }
                // if there is no change, do nothing
                if (ltNode.Equals(prevGridViewSelection))
                {
                    return;
                }
                prevGridViewSelection = ltNode;

                //
                // LT text hint
                //

                richTextBoxLTInfo.Clear();

                const string HTML_DIRECTORY = @"Resources\html";
                string htmlFileName = (ltNode as LearningTaskNode).TaskType.Name + ".html";
                string fullPath = MyResources.GetMyAssemblyPath() + "\\" + HTML_DIRECTORY + "\\" + htmlFileName;


                if (File.Exists(fullPath))
                {
                    // Create a file to write to.
                    string htmlPage = File.ReadAllText(fullPath);

                    string name = System.Text.RegularExpressions.Regex.Match(htmlPage, "<title>.*</title>").ToString();
                    if (name.Length > 0)
                    {
                        name = name.Split('>', '<')[2];
                    }
                    richTextBoxLTInfo.AppendText(name + "\r\n\r\n");
                    richTextBoxLTInfo.SelectAll();
                    richTextBoxLTInfo.SelectionFont = new Font(richTextBoxLTInfo.Font, FontStyle.Bold);

                    string description = System.Text.RegularExpressions.Regex.Match(htmlPage,
                        "Description(.*?)<td(.*?)</td>", System.Text.RegularExpressions.RegexOptions.Singleline).ToString();
                    if (description.Length > 0)
                    {
                        description = description.Split('>', '<')[4];
                    }
                    description = description.Replace(System.Environment.NewLine, "");
                    richTextBoxLTInfo.AppendText(description);
                }

                //
                // LVL tab
                //
                tabControlLevels.TabPages.Clear();

                Type ltType = ltNode.TaskType;
                ILearningTask lt = LearningTaskFactory.CreateLearningTask(ltType);
                TrainingSetHints hints = lt.TSProgression[0];

                Levels = new List<LevelNode>();
                LevelGrids = new List<DataGridView>();
                Attributes = new List<List<AttributeNode>>();
                AttributesChange = new List<List<int>>();

                for (int i = 0; i < lt.TSProgression.Count; i++)
                {
                    // create tab
                    LevelNode ln = new LevelNode(i + 1);
                    Levels.Add(ln);
                    TabPage tp = new TabPage(ln.Text);
                    tabControlLevels.TabPages.Add(tp);

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
                    dgv.CellFormatting += levelGrid_CellFormatting;
                    dgv.SelectionChanged += levelGrid_SelectionChanged;
                    dgv.ClearSelection();

                    tabControlLevels.Update();
                }
            }
            ));
        }

        private void levelGrid_SelectionChanged(object sender, EventArgs a)
        {
            richTextBoxLTLevelInfo.Clear();
            DataGridView dgv = sender as DataGridView;
            int level = LevelGrids.IndexOf(dgv);
            if (dgv.SelectedRows.Count == 0)
            {
                return;
            }
            int row = dgv.SelectedRows[0].Index;
            richTextBoxLTLevelInfo.AppendText(Attributes[level][row].GetAnotation());
        }

        private void levelGrid_CellFormatting(object sender, DataGridViewCellFormattingEventArgs args)
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
            /*int row = args.RowIndex;
            int column = args.ColumnIndex;
            dgv.Rows[row].Cells[column].ToolTipText = Attributes[level][row].GetAnotation();*/
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
            splitContainer2.Panel2Collapsed = !Properties.School.Default.ShowVisual;
            m_emulateSuccess = btnEmulateSuccess.Checked;
            SchoolWorld school = m_mainForm.Project.World as SchoolWorld;
            UpdateWorldHandlers(school, school);

            UpdateWindowName(sender, e);
            UpdateData();
        }

        // /////////////////////////////////////////////////////////////////////////////////// //
        // /////////////////////////////////////////////////////////////////////////////////// //
        // /////////////////////////////////////////////////////////////////////////////////// //
        // ///////////////////////////////// SCHOOL MAIN FORM //////////////////////////////// //
        // /////////////////////////////////////////////////////////////////////////////////// //
        // /////////////////////////////////////////////////////////////////////////////////// //
        // /////////////////////////////////////////////////////////////////////////////////// //

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
            if (!IsWorkspaceSaved)
                Text += '*';
        }

        #region DragDrop

        private void tree_ItemDrag(object sender, System.Windows.Forms.ItemDragEventArgs e)
        {
            treeViewLTList.DoDragDropSelectedNodes(DragDropEffects.Move);
        }

        private void tree_DragOver(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(typeof(TreeNodeAdv[])) && treeViewLTList.DropPosition.Node != null)
            {
                TreeNodeAdv draggedNode = (e.Data.GetData(typeof(TreeNodeAdv[])) as TreeNodeAdv[]).First();
                TreeNodeAdv parent = treeViewLTList.DropPosition.Node;
                if (treeViewLTList.DropPosition.Position != NodePosition.Inside)
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
            treeViewLTList.BeginUpdate();

            TreeNodeAdv[] nodes = (TreeNodeAdv[])e.Data.GetData(typeof(TreeNodeAdv[]));
            if (treeViewLTList.DropPosition.Node == null)
                return;
            Node dropNode = treeViewLTList.DropPosition.Node.Tag as Node;
            if (treeViewLTList.DropPosition.Position == NodePosition.Inside)
            {
                foreach (TreeNodeAdv n in nodes)
                {
                    (n.Tag as Node).Parent = dropNode;
                }
                treeViewLTList.DropPosition.Node.IsExpanded = true;
            }
            else
            {
                Node parent = dropNode.Parent;
                Node nextItem = dropNode;
                if (treeViewLTList.DropPosition.Position == NodePosition.After)
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

            treeViewLTList.EndUpdate();
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
        }

        private void btnNewCurr_Click(object sender, EventArgs e)
        {
            CurriculumNode node = new CurriculumNode { Text = "Curr" + m_model.Nodes.Count.ToString() };
            m_model.Nodes.Add(node);

            // Curriculum name directly editable upon creation
            treeViewLTList.SelectedNode = treeViewLTList.FindNodeByTag(node);
            NodeTextBox control = (NodeTextBox)treeViewLTList.NodeControls.ElementAt(1);
            control.BeginEdit();
        }

        private void btnDelete_Click(object sender, EventArgs e)
        {
            if (treeViewLTList.SelectedNode == null)
            {
                return;
            }
            if (treeViewLTList.SelectedNode.Tag is LearningTaskNode || treeViewLTList.SelectedNode.Tag is CurriculumNode)
            {
                DeleteNodes(sender, e);
                return;
            }
            Node parent = (treeViewLTList.SelectedNode.Tag as Node).Parent;
            if (parent != null && parent is CurriculumNode)
                parent.Parent = null;
        }

        private void btnNewTask_Click(object sender, EventArgs e)
        {
            if (treeViewLTList.SelectedNode == null)
                return;

            AddTaskView = new LearningTaskSelectionForm();
            AddTaskView.StartPosition = FormStartPosition.CenterParent;
            AddTaskView.ShowDialog(this);
            if (AddTaskView.ResultLearningTaskTypes == null)
                return;

            List<LearningTaskNode> newLearningTaskNodes = new List<LearningTaskNode>();
            foreach (Tuple<Type,Type> learningTaskInfo in AddTaskView.ResultLearningTaskTypes)
            {
                newLearningTaskNodes.Add(new LearningTaskNode(learningTaskInfo.Item1, learningTaskInfo.Item2));
            }
            //LearningTaskNode task = new LearningTaskNode(AddTaskView.ResultTaskType, AddTaskView.ResultWorldType);

            if (newLearningTaskNodes.Count > 0)
            {
                if (treeViewLTList.SelectedNode.Tag is CurriculumNode)
                {
                    foreach (LearningTaskNode node in newLearningTaskNodes)
                    {
                        (treeViewLTList.SelectedNode.Tag as Node).Nodes.Add(node);
                    }
                    treeViewLTList.SelectedNode.IsExpanded = true;
                }
                else if (treeViewLTList.SelectedNode.Tag is LearningTaskNode)
                {
                    LearningTaskNode source = treeViewLTList.SelectedNode.Tag as LearningTaskNode;
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
            // Walking through the nodes backwards. That way the index doesn't increase past the
            // node size
            for (int i = treeViewLTList.SelectedNodes.Count - 1; i >= 0; i--)
            {
                // After 1/many nodes are deleted, select the node that was after it/them
                if (i == treeViewLTList.SelectedNodes.Count - 1)
                {
                    TreeNodeAdv nextone = treeViewLTList.SelectedNode.NextNode;
                    if (nextone != null)
                    {
                        nextone.IsSelected = true;
                    }
                }

                TreeNodeAdv n = (TreeNodeAdv)treeViewLTList.SelectedNodes[i];
                (n.Tag as Node).Parent = null;
            }
            if(treeViewLTList.SelectedNodes.Count == 0)
            {
                richTextBoxLTInfo.Clear();
                richTextBoxLTLevelInfo.Clear();
            }
        }

        private void btnAutosave_CheckedChanged(object sender, EventArgs e)
        {
            bool check = (sender as ToolStripButton).Checked;
            Properties.School.Default.AutosaveEnabled = check;
            Properties.School.Default.Save();

            if (!check)
            {
                Properties.School.Default.AutosaveFolder = null;
                Properties.School.Default.Save();
                return;
            }

            if (!String.IsNullOrEmpty(Properties.School.Default.AutosaveFolder))
                return;

            if (folderBrowserAutosave.ShowDialog() != DialogResult.OK)
            {
                (sender as ToolStripButton).CheckState = 0;
                Properties.School.Default.AutosaveEnabled = false;
                Properties.School.Default.Save();
                return;
            }

            Properties.School.Default.AutosaveFolder = folderBrowserAutosave.SelectedPath;
            Properties.School.Default.Save();
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
            if (treeViewLTList.SelectedNode == null)
                return;

            DockContent detailsForm = null;
            if (treeViewLTList.SelectedNode.Tag is CurriculumNode)
            {
                CurriculumNode curr = treeViewLTList.SelectedNode.Tag as CurriculumNode;
                detailsForm = new SchoolCurrDetailsForm(curr);
                detailsForm.Text = curr.Text;
            }
            else if (treeViewLTList.SelectedNode.Tag is LearningTaskNode)
            {
                LearningTaskNode node = treeViewLTList.SelectedNode.Tag as LearningTaskNode;
                detailsForm = new SchoolTaskDetailsForm(node.TaskType);
                detailsForm.Text = node.Text;
            }
            if (detailsForm != null)
                OpenFloatingOrActivate(detailsForm, DockPanel);
        }

        private void btnToggleCheck(object sender, EventArgs e)
        {
            (sender as ToolStripButton).Checked = !(sender as ToolStripButton).Checked;
        }

        #endregion Button clicks

        private void SaveProjectAs(object sender, EventArgs e)
        {
            if (saveFileDialog1.ShowDialog() != DialogResult.OK)
                return;

            // set LOF before saving because after save, window name is updated and that process uses LOF value
            Properties.School.Default.LastOpenedFile = saveFileDialog1.FileName;
            Properties.School.Default.Save();
            SaveProject(saveFileDialog1.FileName);
        }

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

        private void btnSaveResults_Click(object sender, EventArgs e)
        {

            if (dataGridViewLearningTasks.GetClipboardContent() == null)
            {
                MessageBox.Show("There is no data to save, please create at least one task before saving the results", "Nothing to save", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            if (saveResultsDialog.ShowDialog() != DialogResult.OK)
                return;

            ExportDataGridViewData(saveResultsDialog.FileName);
        }

        private void tabControl1_SelectedIndexChanged(object sender, EventArgs e)
        {
            if ((sender as TabControl).SelectedTab == null)
            {
                return;
            }
            ((sender as TabControl).SelectedTab.Controls[0] as DataGridView).ClearSelection();
        }

        private void SchoolRunForm_Load(object sender, EventArgs e)
        {
            // this line has to be in this event - before the SetObserver is called, otherwise
            // first time SetObserver runs, m_showObserver is TRUE even if it is actually supposed to be FALSE
            // because of this, m_observer is set to some invalid stuff AFTER the project loads and m_school world
            // and its memblocks are changed
            // what the hell I was thinking when I put this to VisibleChanged??
            btnObserver.Checked = Properties.School.Default.ShowVisual;
            // TODO: Maybe other stuff from SchoolRunForm_VisibleChanged should go here?
        }
    }
}
