using GoodAI.BrainSimulator.Utils;
using GoodAI.Core;
using GoodAI.Core.Configuration;
using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using GoodAI.BrainSimulator.Properties;
using GoodAI.BrainSimulator.UserSettings;
using GoodAI.Core.Task;
using WeifenLuo.WinFormsUI.Docking;
using YAXLib;

namespace GoodAI.BrainSimulator.Forms
{
    public partial class MainForm : Form
    {        
        private static Color STATUS_BAR_BLUE = Color.FromArgb(255, 0, 122, 204);
        private static Color STATUS_BAR_BLUE_BUILDING = Color.FromArgb(255, 14, 99, 156);

        private MruStripMenuInline m_recentMenu;
        private bool m_isClosing = false;

        public ISet<IMyExecutable> Breakpoints
        {
            get
            {
                if (SimulationHandler != null && SimulationHandler.Simulation != null)
                    return SimulationHandler.Simulation.Breakpoints;

                return null;
            }
        }

        private void MainForm_Load(object sender, EventArgs e)
        {
            UpgradeUserSettings();

            ToolBarNodes.InitDefaultToolBar(Settings.Default, MyConfiguration.KnownNodes);

            if (!TryRestoreViewsLayout(UserLayoutFileName))
            {
                ResetViewsLayout();
            }

            this.WindowState = FormWindowState.Maximized;
            statusStrip.BackColor = STATUS_BAR_BLUE;

            if (!TryOpenStartupProject())
            {
                OpenGraphLayout(Project.Network);
            }

            m_recentMenu = new MruStripMenuInline(fileToolStripMenuItem, recentFilesMenuItem , RecentFiles_Click, 5);

            Project.Restore();

            StringCollection recentFilesList = Settings.Default.RecentFilesList;

            if (recentFilesList != null)
            {
                string[] tmp = new string[recentFilesList.Count];
                recentFilesList.CopyTo(tmp, 0);
                m_recentMenu.AddFiles(tmp);
            }
        }

        private static void UpgradeUserSettings()
        {
            Settings settings = Settings.Default;

            if (!settings.ShouldUpgradeSettings)
                return;

            try
            {
                settings.Upgrade();
                settings.ShouldUpgradeSettings = false;
                settings.Save();

                MyLog.INFO.WriteLine("Settings upgraded.");
            }
            catch (Exception e)
            {
                MyLog.ERROR.WriteLine("Error upgrading settings: " + e.Message);
            }
        }

        private bool TryOpenStartupProject()
        {
            try
            {
                if (!string.IsNullOrEmpty(MyConfiguration.OpenOnStartupProjectName))
                {
                    OpenProject(MyConfiguration.OpenOnStartupProjectName);
                }
                else if (!string.IsNullOrEmpty(Settings.Default.LastProject))
                {
                    OpenProject(Settings.Default.LastProject);
                }
                else
                {
                    return false;
                }
            }
            catch (ProjectLoadingException)  // already logged
            {
                return false;
            }
            catch (Exception ex)
            {
                MyLog.ERROR.WriteLine("Error setting up startup project: " + ex.Message);
                return false;
            }

            return true;
        }

        //TODO: this should be done by data binding but menu items cannot do that (add this support)
        void SimulationHandler_StateChanged(object sender, MySimulationHandler.StateEventArgs e)
        {
            runToolButton.Enabled = SimulationHandler.CanStart;            
            startToolStripMenuItem.Enabled = SimulationHandler.CanStart;            

            pauseToolButton.Enabled = SimulationHandler.CanPause;
            pauseToolStripMenuItem.Enabled = SimulationHandler.CanPause;

            stopToolButton.Enabled = SimulationHandler.CanStop;
            stopToolStripMenuItem.Enabled = SimulationHandler.CanStop;


            debugToolButton.Enabled = SimulationHandler.CanStartDebugging;
            debugToolStripMenuItem.Enabled = SimulationHandler.CanStartDebugging;

            stepOverToolButton.Enabled = SimulationHandler.CanStepOver;
            stepOverToolStripMenuItem.Enabled = SimulationHandler.CanStepOver;

            stepIntoToolStripMenuItem.Enabled = SimulationHandler.CanStepInto;
            stepOutToolStripMenuItem.Enabled = SimulationHandler.CanStepOut;

            reloadButton.Enabled = SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED;

            simStatusLabel.Text = SimulationHandler.State.GetAttributeProperty((DescriptionAttribute x) => x.Description);           

            //TODO: this is awful, binding is needed here for sure            
            newProjectToolButton.Enabled = SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED;
            newProjectToolStripMenuItem.Enabled = SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED;

            openProjectToolButton.Enabled = SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED;
            openProjectToolStripMenuItem.Enabled = SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED;

            saveProjectToolButton.Enabled = SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED;
            saveProjectAsToolStripMenuItem.Enabled = SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED;
            saveProjectToolStripMenuItem.Enabled = SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED;

            copySelectionToolStripMenuItem.Enabled = pasteSelectionToolStripMenuItem.Enabled =
                SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED;

            worldList.Enabled = SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED;

            NodePropertyView.CanEdit = SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED;

            updateMemoryBlocksToolStripMenuItem.Enabled = SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED;
            
            MemoryBlocksView.Enabled = SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED ||
                 SimulationHandler.State == MySimulationHandler.SimulationState.PAUSED;

            if (SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED)
            {                
                stepStatusLabel.Text = String.Empty;
                statusStrip.BackColor = STATUS_BAR_BLUE;

                exportStateButton.Enabled = MyMemoryBlockSerializer.TempDataExists(Project);
                clearDataButton.Enabled = exportStateButton.Enabled;
            }
            else if (SimulationHandler.State == MySimulationHandler.SimulationState.PAUSED)
            {
                statusStrip.BackColor = Color.Chocolate;
            }
            else
            {
                statusStrip.BackColor = STATUS_BAR_BLUE_BUILDING;
            }
            RefreshUndoRedoButtons();
        }    

        private void openProjectToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (openFileDialog.ShowDialog(this) == DialogResult.OK)
            {
                OpenProjectAndAddToRecentMenu(openFileDialog.FileName);
            }            
        }

        private void OpenProjectAndAddToRecentMenu(string fileName)
        {
            try
            {
                OpenProject(fileName);
                m_recentMenu.AddFile(fileName);
            }
            catch (ProjectLoadingException)
            {
                // already logged
            }
            catch (Exception ex)
            {
                MyLog.ERROR.WriteLine("Error while opening a project:" + ex.Message);
            }
        }

        private void nodeSettingsMenuItem_Click(object sender, EventArgs e)
        {
            NodeSelectionForm selectionForm = new NodeSelectionForm(this);
            selectionForm.StartPosition = FormStartPosition.CenterParent;

            if (selectionForm.ShowDialog(this) == DialogResult.OK)
            {
                PopulateWorldList();

                foreach (GraphLayoutForm gv in GraphViews.Values)
                {
                    gv.InitToolBar();
                }
            }
        }

        private void viewToolStripMenuItem_Click(object sender, EventArgs e)
        {
            DockContent view = (sender as ToolStripMenuItem).Tag as DockContent;
            OpenFloatingOrActivate(view);
        }

        private void OpenFloatingOrActivate(DockContent view)
        {
            if ((view.DockAreas & DockAreas.Float) > 0 && !view.Created)
            {
                Size viewSize = new Size(view.Bounds.Size.Width, view.Bounds.Size.Height);
                view.Show(dockPanel, DockState.Float);
                view.FloatPane.FloatWindow.Size = viewSize;                
            }
            else
            {
                view.Activate();
            }
        }

        public void OpenNodeHelpView()
        {
            OpenFloatingOrActivate(HelpView);            
        }

        private void saveProjectToolStripMenuItem_Click(object sender, EventArgs e)
        {
            SaveProjectOrSaveAs();
        }

        private void SaveProjectOrSaveAs()
        {
            if (saveFileDialog.FileName != string.Empty)
            {
                SaveProject(saveFileDialog.FileName);
            }
            else
            {
                SaveProjectAs();  // ask for file name and then save the project
            }
        }

        private void saveProjectAsToolStripMenuItem_Click(object sender, EventArgs e)
        {
            SaveProjectAs();
        }

        private void SaveProjectAs()
        {
            if (saveFileDialog.ShowDialog(this) != DialogResult.OK)
                return;

            string newName = saveFileDialog.FileName;

            string oldProjectDataPath = MyMemoryBlockSerializer.GetTempStorage(Project);
            string newProjectDataPath = MyMemoryBlockSerializer.GetTempStorage(MyProject.MakeNameFromPath(newName));

            if (newProjectDataPath != oldProjectDataPath)
                CopyDirectory(oldProjectDataPath, newProjectDataPath);
            else
                MyLog.WARNING.WriteLine("Projects with the same filename share the same temporal folder where the state is saved.");

            Project.SetNameFromPath(newName);

            SaveProject(newName);
            m_recentMenu.AddFile(newName);
        }

        private void CopyDirectory(string sourcePath, string destinationPath)
        {
            if (!Directory.Exists(sourcePath) || (sourcePath == destinationPath))
                return;

            try
            {
                // Create all of the directories.
                foreach (string dirPath in Directory.GetDirectories(sourcePath, "*", SearchOption.AllDirectories))
                    Directory.CreateDirectory(dirPath.Replace(sourcePath, destinationPath));

                // Copy all the files & replace any files with the same name.
                foreach (string sourceFilePath in Directory.GetFiles(sourcePath, "*.*", SearchOption.AllDirectories))
                    File.Copy(sourceFilePath, sourceFilePath.Replace(sourcePath, destinationPath), true);
            }
            catch (Exception ex)
            {
                MyLog.ERROR.WriteLine("Failed to copy directory: " + ex.Message);
            }
        }

        private void quitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void newProjectToolStripMenuItem_Click(object sender, EventArgs e)
        {
            CloseCurrentProjectWindows();
            CreateNewProject();            

            CreateNetworkView();
            OpenGraphLayout(Project.Network);

            AppSettings.SaveSettings(settings => settings.LastProject = String.Empty);
        }

        public void runToolButton_Click(object sender, EventArgs e)
        {
            ShowHideAllObservers(forceShow: true);
            StartSimulation();            
        }

        private void SetupDebugViews()
        {
            ShowHideAllObservers(forceShow: true);
        }

        public void stopToolButton_Click(object sender, EventArgs e)
        {
            ShowHideAllObservers(forceShow: true);
            SimulationHandler.StopSimulation();
            SimulationHandler.Simulation.InDebugMode = false;
        }

        public void pauseToolButton_Click(object sender, EventArgs e)
        {
            ShowHideAllObservers(forceShow: true);
            SimulationHandler.PauseSimulation();
        }

        private void worldList_SelectedIndexChanged(object sender, EventArgs e)
        {
            CloseObservers(Project.World);

            if (worldList.SelectedItem != null )
            {
                MyWorldConfig wc = worldList.SelectedItem as MyWorldConfig;

                if (Project.World == null || wc.NodeType != Project.World.GetType())
                {
                    var oldWorld = Project.World;

                    Project.CreateWorld(wc.NodeType);
                    Project.World.EnableDefaultTasks();
                    NodePropertyView.Target = null;

                    if (NetworkView != null)
                    {
                        NetworkView.ReloadContent();
                    }

                    foreach (GraphLayoutForm graphView in GraphViews.Values)
                    {
                        graphView.Desktop.Invalidate();                        
                        graphView.worldButton_Click(sender, e);                     
                    }

                    ProjectStateChanged("World selected");

                    if (WorldChanged != null)
                        WorldChanged(this, new WorldChangedEventArgs(oldWorld, Project.World));
                }
            }
        }

        private void MainForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (m_isClosing) return;

            // Cancel the event - the window will close when the simulation is finished.
            e.Cancel = true;
           
            if (SimulationHandler.State != MySimulationHandler.SimulationState.STOPPED)
            {
                DialogResult dialogResult = DialogResult.None;

                PauseSimulationForAction(() =>
                {
                    dialogResult =
                        MessageBox.Show(
                            "Do you want to quit while the simulation is running?",
                            "Quit?",
                            MessageBoxButtons.YesNo, MessageBoxIcon.Question);

                    return dialogResult == DialogResult.No;                    
                });

                if (dialogResult == DialogResult.No)
                {
                    return;
                }
            }           

            if ((String.IsNullOrEmpty(saveFileDialog.FileName)) || !IsProjectSaved(saveFileDialog.FileName))
            {
                var dialogResult = MessageBox.Show("Save project changes?",
                    "Save Changes", MessageBoxButtons.YesNoCancel, MessageBoxIcon.Question);

                // Do not close.
                if (dialogResult == DialogResult.Cancel)
                    return;

                if (dialogResult == DialogResult.Yes)
                    SaveProjectOrSaveAs();
            }

            // When this is true, the event will just return next time it's called.
            m_isClosing = true;
            SimulationHandler.Finish(Close);
        }

        private void MainForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            StoreViewsLayout(UserLayoutFileName);

            AppSettings.SaveSettings(settings =>
            {
                settings.RecentFilesList = new StringCollection();
                settings.RecentFilesList.AddRange(m_recentMenu.GetFiles());
            });
        }

        private void reloadButton_Click(object sender, EventArgs e)
        {
            MyKernelFactory.Instance.ClearLoadedKernels();
            MyLog.INFO.WriteLine("Kernel cache cleared.");
        }

        private void loadUserNodesToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (openNodeFileDialog.ShowDialog(this) == DialogResult.OK)
            {
                AppSettings.SaveSettings(settings => settings.UserNodesFile = openNodeFileDialog.FileName);

                if (MessageBox.Show("Restart is needed for this action to take effect.\nDo you want to quit application?", "Restart needed",
                    MessageBoxButtons.OKCancel, MessageBoxIcon.Question) == DialogResult.OK)
                {
                    Close();
                }
            } 
        }

        private void importProjectToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (openFileDialog.ShowDialog(this) == DialogResult.OK)
            {
                saveFileDialog.FileName = openFileDialog.FileName;
                var dr = MessageBox.Show("Import observers?", "Importing project", MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                ImportProject(openFileDialog.FileName, dr == DialogResult.Yes);
            }            
        }

        private void copySelectionToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (dockPanel.ActiveContent is TextEditForm)
            {
                (dockPanel.ActiveDocument as TextEditForm).CopyText();
            }
            else if (dockPanel.ActiveContent is ConsoleForm)
            {
                Clipboard.SetText((dockPanel.ActiveContent as ConsoleForm).textBox.SelectedText);
                return;
            }
            else if (dockPanel.ActiveDocument is GraphLayoutForm)
            {
                CopySelectedNodesToClipboard();
            }
        }

        private void pasteSelectionToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (dockPanel.ActiveContent is TextEditForm)
            {
                (dockPanel.ActiveDocument as TextEditForm).PasteText();
            }            
            else if (dockPanel.ActiveDocument is GraphLayoutForm)
            {
                PasteNodesFromClipboard();
            }            
        }

        private void RecentFiles_Click(int number, string fileName)
        {
            OpenProjectAndAddToRecentMenu(fileName);
        }

        private void setGlobalDataFolderToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (openMemFileDialog.ShowDialog(this) == DialogResult.OK)
            {
                string dataFolder = MyProject.MakeDataFolderFromFileName(openMemFileDialog.FileName);

                SimulationHandler.Simulation.GlobalDataFolder = dataFolder;
                setGlobalDataFolderToolStripMenuItem.Text = "Change global data folder: " + SimulationHandler.Simulation.GlobalDataFolder;
                clearGlobalDataFolderToolStripMenuItem.Visible = true;
            }
        }

        private void clearGlobalDataFolderToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (MessageBox.Show("Unset the global data folder: " + SimulationHandler.Simulation.GlobalDataFolder + " ?",
                "Unset", MessageBoxButtons.OKCancel, MessageBoxIcon.Question) == DialogResult.OK)
            {
                SimulationHandler.Simulation.GlobalDataFolder = String.Empty;
                setGlobalDataFolderToolStripMenuItem.Text = "Set global data folder";
                clearGlobalDataFolderToolStripMenuItem.Visible = false;
            }
        }

        private void loadOnStartMenuItem_Click(object sender, EventArgs e)
        {
            loadMemBlocksButton.Checked = !loadMemBlocksButton.Checked;

            Project.LoadAllNodesData = loadMemBlocksButton.Checked;
       }

        private void saveOnStopMenuItem_CheckChanged(object sender, EventArgs e)
        {
            Project.SaveAllNodesData = saveMemBlocksButton.Checked;
        }

        private void exportStateButton_Click(object sender, EventArgs e)
        {
            if (saveMemFileDialog.ShowDialog(this) == DialogResult.OK)
            {
                try
                {
                    string dataFolder = MyProject.MakeDataFolderFromFileName(saveMemFileDialog.FileName);
                    
                    MyMemoryBlockSerializer.ExportTempStorage(Project, dataFolder);

                    MyNetworkState networkState = new MyNetworkState()
                    {
                        ProjectName = Project.Name,
                        MemoryBlocksLocation = dataFolder,
                        SimulationStep = SimulationHandler.SimulationStep
                    };

                    YAXSerializer serializer = new YAXSerializer(typeof(MyNetworkState),
                        YAXExceptionHandlingPolicies.ThrowErrorsOnly,
                        YAXExceptionTypes.Warning);

                    serializer.SerializeToFile(networkState, saveMemFileDialog.FileName);
                    MyLog.INFO.WriteLine("Saving state: " + saveMemFileDialog.FileName);
                }
                catch (Exception ex)
                {
                    MyLog.ERROR.WriteLine("Saving state failed: " + ex.Message);
                }    
            }
        }

        private void clearDataButton_Click(object sender, EventArgs e)
        {
            if (MessageBox.Show("Clear all temporal data for project: " + Project.Name + "?", 
                "Clear data", MessageBoxButtons.OKCancel, MessageBoxIcon.Question) == DialogResult.OK)
            {
                MyMemoryBlockSerializer.ClearTempStorage(Project);

                exportStateButton.Enabled = false;
                clearDataButton.Enabled = false;
            }
        }

        private void updateMemoryBlocksToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                SimulationHandler.UpdateMemoryModel();
            }
            finally
            {
                foreach (GraphLayoutForm graphView in GraphViews.Values)
                {
                    graphView.Desktop.Invalidate();
                }
            }
        }

        private void guideToolStripMenuItem_Click(object sender, EventArgs e) 
        {
            try
            {
                MyDocProvider.Navigate(Settings.Default.HelpUrl);
            }
            catch (Exception exc)
            {
                MyLog.ERROR.WriteLine("Failed to get HelpUrl setting: " + exc.Message);
            }
        }

        private void autosaveButton_CheckedChanged(object sender, EventArgs e)
        {
            autosaveTextBox.Enabled = autosaveButton.Checked;
            SimulationHandler.AutosaveEnabled = autosaveButton.Checked;

            AppSettings.SaveSettings(settings => settings.AutosaveEnabled = autosaveButton.Checked);
        }        

        private void autosaveTextBox_Validating(object sender, CancelEventArgs e)
        {
            int result = 0;

            if (int.TryParse(autosaveTextBox.Text, out result))
            {
                SimulationHandler.AutosaveInterval = result;

                AppSettings.SaveSettings(settings => settings.AutosaveInterval = result);
            }
            else
            {
                autosaveTextBox.Text = SimulationHandler.AutosaveInterval + "";
            }
        }

        public void debugToolButton_Click(object sender, EventArgs e)
        {            
            SimulationHandler.Simulation.InDebugMode = true;
            StartSimulationStep();

            OpenFloatingOrActivate(DebugView);        
        }

        public void stepOverToolButton_Click(object sender, EventArgs e)
        {
            SetupDebugViews();

            SimulationHandler.Simulation.StepOver();
            if (SimulationHandler.Simulation.InDebugMode)
            {
                // In debug mode, the simulation always runs - it is stopped internally by what is set in StepOver
                // and similar methods.
                StartSimulation();
            }
            else
            {
                StartSimulationStep();
            }

        }

        private void stepIntoToolStripMenuItem_Click(object sender, EventArgs e)
        {
            SetupDebugViews();

            SimulationHandler.Simulation.StepInto();
            StartSimulation();
        }

        private void stepOutToolStripMenuItem_Click(object sender, EventArgs e)
        {
            SetupDebugViews();

            SimulationHandler.Simulation.StepOut();
            StartSimulation();
        }

        private void aboutToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var aboutDialog = new AboutDialog();
            aboutDialog.ShowDialog();
        }

        private bool handleFirstClickOnActivated = false;

        /// <summary>
        /// Raises the <see cref="E:System.Windows.Forms.Form.Activated" /> event.
        /// Handle WinForms bug for first click during activation
        /// </summary>
        /// <param name="e">An <see cref="T:System.EventArgs" /> that contains the event data.</param>
        protected override void OnActivated(EventArgs e)
        {
            base.OnActivated(e);
            if (this.handleFirstClickOnActivated)
            {
                var cursorPosition = Cursor.Position;
                var clientPoint = this.PointToClient(cursorPosition);
                var child = this.GetChildAtPoint(clientPoint);

                while (this.handleFirstClickOnActivated && child != null)
                {
                    var toolStrip = child as ToolStrip;
                    if (toolStrip != null)
                    {
                        this.handleFirstClickOnActivated = false;
                        clientPoint = toolStrip.PointToClient(cursorPosition);
                        foreach (var item in toolStrip.Items)
                        {
                            var toolStripItem = item as ToolStripItem;
                            if (toolStripItem != null && toolStripItem.Bounds.Contains(clientPoint))
                            {
                                var tsMenuItem = item as ToolStripDropDownItem;
                                if (tsMenuItem != null)
                                {
                                    tsMenuItem.ShowDropDown();
                                    break;
                                }

                                toolStripItem.PerformClick();
                                break;
                            }
                        }
                    }
                    else
                    {
                        child = child.GetChildAtPoint(clientPoint);
                    }
                }
                this.handleFirstClickOnActivated = false;
            }
        }

        /// <summary>
        /// If the form is being focused (activated), set the handleFirstClickOnActivated flag
        /// indicating that so that it can be later used in OnActivated.
        /// </summary>
        /// <param name="m"></param>
        protected override void WndProc(ref Message m)
        {
            const int WM_ACTIVATE = 0x0006;
            const int WA_CLICKACTIVE = 0x0002;
            if (m.Msg == WM_ACTIVATE && Low16(m.WParam) == WA_CLICKACTIVE)
            {
                handleFirstClickOnActivated = true;
            }
            base.WndProc(ref m);
        }

        private static int GetIntUnchecked(IntPtr value)
        {
            return IntPtr.Size == 8 ? unchecked((int)value.ToInt64()) : value.ToInt32();
        }

        private static int Low16(IntPtr value)
        {
            return unchecked((short)GetIntUnchecked(value));
        }

        private static int High16(IntPtr value)
        {
            return unchecked((short)(((uint)GetIntUnchecked(value)) >> 16));
        }

        private void profileToolButton_CheckedChanged(object sender, EventArgs e)
        {
            MyExecutionBlock.IsProfiling = profileToolButton.Checked;
        }

        // Using MouseUp instead of Click because the .Enabled property is updated as part of the action
        // which would sometimes lead to double-clicks.
        private void undoButton_MouseUp(object sender, MouseEventArgs e)
        {
            Undo();
        }

        private void redoButton_MouseUp(object sender, MouseEventArgs e)
        {
            Redo();
        }

        private void undoToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Undo();
        }

        private void redoToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Redo();
        }
    }
}