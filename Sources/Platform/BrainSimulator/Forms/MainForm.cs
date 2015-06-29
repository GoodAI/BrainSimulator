using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using BrainSimulatorGUI.NodeView;
using Graph;
using Graph.Compatibility;
using System.IO;
using WeifenLuo.WinFormsUI.Docking;
using BrainSimulator.Nodes;
using BrainSimulator;
using System.Diagnostics;
using BrainSimulator.Utils;
using BrainSimulatorGUI.Utils;
using BrainSimulator.Observers;
using BrainSimulatorGUI.Nodes;
using System.Reflection;
using System.Collections.Specialized;
using BrainSimulator.Execution;
using YAXLib;
using BrainSimulator.Memory;
using BrainSimulator.Configuration;

namespace BrainSimulatorGUI.Forms
{
    public partial class MainForm : Form
    {        
        private static Color STATUS_BAR_BLUE = Color.FromArgb(255, 0, 122, 204);
        private static Color STATUS_BAR_BLUE_BUILDING = Color.FromArgb(255, 14, 99, 156);

        private MruStripMenuInline m_recentMenu;

        private void MainForm_Load(object sender, EventArgs e)
        {
            FileInfo viewLayoutFile = new FileInfo(UserLayoutFileName);

            if (viewLayoutFile.Exists)
            {
                RestoreViewsLayout(viewLayoutFile.FullName);
            }
            else
            {
                ResetViewsLayout();
            }

            this.WindowState = FormWindowState.Maximized;
            statusStrip.BackColor = STATUS_BAR_BLUE;

            if (!string.IsNullOrEmpty(MyConfiguration.OpenOnStartupProjectName))
            {
                if (!OpenProject(MyConfiguration.OpenOnStartupProjectName))
                {
                    OpenGraphLayout(Project.Network);
                }            
            }
            else if (!string.IsNullOrEmpty(Properties.Settings.Default.LastProject))
            {
                if (!OpenProject(Properties.Settings.Default.LastProject))
                {
                    OpenGraphLayout(Project.Network);
                }
            }
            else
            {
                OpenGraphLayout(Project.Network);
            }            

            m_recentMenu = new MruStripMenuInline(fileToolStripMenuItem, recentFilesMenuItem , RecentFiles_Click, 5);

            StringCollection recentFilesList = Properties.Settings.Default.RecentFilesList;

            if (recentFilesList != null)
            {
                string[] tmp = new string[recentFilesList.Count];
                recentFilesList.CopyTo(tmp, 0);
                m_recentMenu.AddFiles(tmp);
            }
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
        }    

        private void openProjectToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (openFileDialog.ShowDialog(this) == DialogResult.OK)
            {
                saveFileDialog.FileName = openFileDialog.FileName;
                OpenProject(openFileDialog.FileName);
                m_recentMenu.AddFile(openFileDialog.FileName);
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
            AskForFileNameAndSaveProject();
        }

        private void AskForFileNameAndSaveProject()
        {
            if (saveFileDialog.FileName != string.Empty)
            {
                SaveProject(saveFileDialog.FileName);
            }
            else
            {
                if (saveFileDialog.ShowDialog(this) == DialogResult.OK)
                {
                    SaveProject(saveFileDialog.FileName);
                    m_recentMenu.AddFile(saveFileDialog.FileName);
                }
            }
        }

        private void saveProjectAsToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (saveFileDialog.ShowDialog(this) == DialogResult.OK)
            {
                SaveProject(saveFileDialog.FileName);
                m_recentMenu.AddFile(saveFileDialog.FileName);
            }
        }

        private void quitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void newProjectToolStripMenuItem_Click(object sender, EventArgs e)
        {
            CloseAllGraphLayouts();
            CloseAllObservers();

            CreateNewProject();            

            CreateNetworkView();
            OpenGraphLayout(Project.Network);

            Properties.Settings.Default.LastProject = String.Empty;
            saveFileDialog.FileName = String.Empty;
        }

        private void runToolButton_Click(object sender, EventArgs e)
        {
            ConsoleView.Activate();            
            StartSimulation(false);            
        }

        private void stepOverToolButton_Click(object sender, EventArgs e)
        {
            ConsoleView.Activate();
            StartSimulation(true);            
        }

        private void stopToolButton_Click(object sender, EventArgs e)
        {
            SimulationHandler.StopSimulation();
            SimulationHandler.Simulation.InDebugMode = false;
        }

        private void pauseToolButton_Click(object sender, EventArgs e)
        {         
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
                    Project.CreateWorld(wc.NodeType);
                    Project.World.EnableAllTasks();
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
                }
            }
        }

        private void MainForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            if ((String.IsNullOrEmpty(saveFileDialog.FileName)) || !IsProjectSaved(saveFileDialog.FileName))
            {
                var dialogResult = MessageBox.Show("Save project changes?",
                    "Save Changes", MessageBoxButtons.YesNoCancel, MessageBoxIcon.Question);

                if (dialogResult == DialogResult.Yes)
                {
                    AskForFileNameAndSaveProject();
                }
                else if (dialogResult == DialogResult.Cancel)
                {
                    e.Cancel = true;
                    return;
                }
            }
        }

        private void MainForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            StoreViewsLayout(UserLayoutFileName);

            Properties.Settings.Default.RecentFilesList = new StringCollection();
            Properties.Settings.Default.RecentFilesList.AddRange(m_recentMenu.GetFiles());

            Properties.Settings.Default.Save();

            SimulationHandler.Finish();
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
                Properties.Settings.Default.UserNodesFile = openNodeFileDialog.FileName;

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
            CopySelectedNodesToClipboard();            
        }

        private void pasteSelectionToolStripMenuItem_Click(object sender, EventArgs e)
        {
            PasteNodesFromClipboard();
        }

        private void RecentFiles_Click(int number, string filename)
        {
            saveFileDialog.FileName = filename;
            OpenProject(filename);
            m_recentMenu.AddFile(saveFileDialog.FileName);
        }        

        private void loadOnStartMenuItem_Click(object sender, EventArgs e)
        {
            if (!loadMemBlocksButton.Checked && openMemFileDialog.ShowDialog(this) == DialogResult.OK)
            {
                string dataFolder = Path.GetDirectoryName(openMemFileDialog.FileName) + "\\" +
                        Path.GetFileNameWithoutExtension(openMemFileDialog.FileName) + ".statedata";

                SimulationHandler.Simulation.GlobalDataFolder = dataFolder;
                SimulationHandler.Simulation.LoadAllNodesData = true;
                loadMemBlocksButton.Checked = true;
            }
            else
            {
                SimulationHandler.Simulation.LoadAllNodesData = false;
                SimulationHandler.Simulation.GlobalDataFolder = String.Empty;
                loadMemBlocksButton.Checked = false;                
            }
        }

        private void saveOnStopMenuItem_CheckChanged(object sender, EventArgs e)
        {
            SimulationHandler.Simulation.SaveAllNodesData = saveMemBlocksButton.Checked;
        }

        private void exportStateButton_Click(object sender, EventArgs e)
        {
            if (saveMemFileDialog.ShowDialog(this) == DialogResult.OK)
            {
                try
                {
                    string dataFolder = Path.GetDirectoryName(saveMemFileDialog.FileName) + "\\" +
                        Path.GetFileNameWithoutExtension(saveMemFileDialog.FileName) + ".statedata";
                    
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
            UpdateMemoryModel();

            foreach (GraphLayoutForm graphView in GraphViews.Values)
            {
                graphView.Desktop.Invalidate();
            }
        }

        private void guideToolStripMenuItem_Click(object sender, EventArgs e) 
        {
            MyDocProvider.Navigate(@"file:///D:/KeenSWH/AI/Docs/BrainSimulator%20Guide/site/index.html");
        }

        private void autosaveButton_CheckedChanged(object sender, EventArgs e)
        {
            autosaveTextBox.Enabled = autosaveButton.Checked;
            SimulationHandler.AutosaveEnabled = autosaveButton.Checked;
            Properties.Settings.Default.AutosaveEnabled = autosaveButton.Checked;
        }        

        private void autosaveTextBox_Validating(object sender, CancelEventArgs e)
        {
            int result = 0;

            if (int.TryParse(autosaveTextBox.Text, out result))
            {
                SimulationHandler.AutosaveInterval = result;
                Properties.Settings.Default.AutosaveInterval = result;
            }
            else
            {
                autosaveTextBox.Text = SimulationHandler.AutosaveInterval + "";
            }
        }

        private void debugToolButton_Click(object sender, EventArgs e)
        {            
            SimulationHandler.Simulation.InDebugMode = true;
            StartSimulation(true);

            OpenFloatingOrActivate(DebugView);        
        }

        private void stepIntoToolStripMenuItem_Click(object sender, EventArgs e)
        {
            StartSimulation(true);
        }

        private void stepOutToolStripMenuItem_Click(object sender, EventArgs e)
        {
            StartSimulation(true);
        }    
    }
}