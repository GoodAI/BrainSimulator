using GoodAI.BrainSimulator.NodeView;
using GoodAI.BrainSimulator.Properties;
using GoodAI.BrainSimulator.UserSettings;
using GoodAI.BrainSimulator.Utils;
using GoodAI.Core.Configuration;
using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers;
using GoodAI.Core.Project;
using GoodAI.Core.Utils;
using GoodAI.Platform.Core.Utils;
using Graph;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Design;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;
using GoodAI.BrainSimulator.Properties;
using GoodAI.BrainSimulator.UserSettings;
using GoodAI.Platform.Core.Utils;
using GoodAI.TypeMapping;
using WeifenLuo.WinFormsUI.Docking;
using YAXLib;

namespace GoodAI.BrainSimulator.Forms
{

    public partial class MainForm : Form
    {
        private static string TITLE_TEXT = "Brain Simulator";

        public MySimulationHandler SimulationHandler { get; private set; }
        public MyDocProvider Documentation { get; private set; }

        public UndoManager UndoManager { get; set; }

        public class WorldChangedEventArgs : EventArgs
        {
            public WorldChangedEventArgs(MyWorld oldWorld, MyWorld newWorld)
            {
                OldWorld = oldWorld;
                NewWorld = newWorld;
            }

            public MyWorld OldWorld { get; private set; }
            public MyWorld NewWorld { get; private set; }
        }

        public event EventHandler<WorldChangedEventArgs> WorldChanged;

        #region Project

        private MyProject m_project;

        public MyProject Project
        {
            get { return m_project; }
            private set
            {
                if (m_project != null)
                {
                    m_project.Dispose();
                }
                m_project = value;
                SimulationHandler.Project = value;
                m_project.SimulationHandler = SimulationHandler;
            }
        }

        private string m_savedProjectRepresentation = null;

        private void CreateNewProject()
        {
            var oldProject = Project;

            Project = new MyProject();
            Project.Network = Project.CreateNode<MyNetwork>();
            Project.Network.Name = "Network";

            string tmpPath = Path.GetTempFileName();
            Project.Name = MyProject.MakeNameFromPath(tmpPath);

            worldList.SelectedIndex = -1;
            worldList.SelectedItem = MyConfiguration.KnownWorlds.Values.First();

            Text = TITLE_TEXT + " - New Project";

            exportStateButton.Enabled = false;
            clearDataButton.Enabled = false;

            Project.Restore();
            RestoreDashboardForm();

            UndoManager.Clear();
            SaveState(GetSerializedProject(tmpPath), tmpPath, "New project");
            RefreshUndoRedoButtons();

            ApplyProjectOptions();  // Apply default project options such as save on stop to UI.

            saveFileDialog.FileName = string.Empty;
        }

        private void CloseCurrentProjectWindows()
        {
            SuppressStateSaving = true;
            CloseAllGraphLayouts();
            CloseAllTextEditors();
            CloseAllObservers();
            SuppressStateSaving = false;
        }

        private void SaveProject(string fileName)
        {
            MyLog.INFO.WriteLine("Saving project: " + fileName);
            try
            {
                string fileContent = GetSerializedProject(fileName);

                ProjectLoader.SaveProject(fileName, fileContent,
                    MyMemoryBlockSerializer.GetTempStorage(Project));

                m_savedProjectRepresentation = fileContent;

                AppSettings.SaveSettings(settings => settings.LastProject = fileName);

                Text = TITLE_TEXT + " - " + Project.Name;
            }
            catch (Exception e)
            {
                MyLog.ERROR.WriteLine("Project saving failed: " + e.Message);
            }
        }

        // TODO(Premek): move this exception inside project loader
        private class ProjectLoadingException : Exception
        {
            public ProjectLoadingException(string message, Exception innerException)
                : base(message, innerException)
            { }
        }

        private void OpenProject(string fileName)
        {
            MyLog.INFO.WriteLine("--------------");
            MyLog.INFO.WriteLine("Loading project: " + fileName);

            string content;

            try
            {
                string newProjectName = Path.GetFileNameWithoutExtension(fileName);

                content = ProjectLoader.LoadProject(fileName,
                    MyMemoryBlockSerializer.GetTempStorage(newProjectName));

                LoadSerializedContent(content, fileName, newProjectName);

                UndoManager.Clear();
                UndoManager.SaveState(new ProjectState(content)
                {
                    ProjectPath = fileName,
                    Action = "Project opened"
                });
                RefreshUndoRedoButtons();
            }
            catch (Exception e)
            {
                MyLog.ERROR.WriteLine("Project loading failed: " + e.Message);
                throw new ProjectLoadingException("Project loading failed.", e);
            }

            AppSettings.SaveSettings(settings => settings.LastProject = fileName);

            saveFileDialog.FileName = fileName;

            m_savedProjectRepresentation = content;  // for "needs saving" detection

            exportStateButton.Enabled = MyMemoryBlockSerializer.TempDataExists(Project);
            clearDataButton.Enabled = exportStateButton.Enabled;

            Text = TITLE_TEXT + " - " + Project.Name;
        }

        private void ImportProject(string fileName, bool showObservers = false)
        {
            MyLog.INFO.WriteLine("Importing project: " + fileName);

            try
            {
                string dataStoragePath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());

                string content = ProjectLoader.LoadProject(fileName, dataStoragePath);

                // data are not used yet, delete them to prevent littering the file system
                // TODO(Premek): Also import (=copy) trained data from the dataStoragePath (?)
                // https://app.asana.com/0/32893784542247/56397605644507
                if (fileName.EndsWith(".brainz"))  // temp directory is only used for brainz
                    Directory.Delete(dataStoragePath, recursive: true);

                // Do not restore links here - that would automatically restore observers and dashboard.
                MyProject importedProject = MyProject.Deserialize(content, Path.GetDirectoryName(fileName),
                    restoreModelOnly: true);

                //offset all imported nodes
                float maxY = NetworkView.Desktop.GetContentBounds().Bottom;
                foreach (var node in importedProject.Network.Children)
                {
                    node.Location.Y += maxY + 10.0f;
                }

                if (showObservers && importedProject.Observers != null)
                    importedProject.RestoreObservers();

                Project.Network.AppendGroupContent(importedProject.Network);

                if (showObservers && importedProject.Observers != null)
                {
                    foreach (MyAbstractObserver observer in importedProject.Observers)
                    {
                        observer.UpdateTargetIdentifier();
                    }
                }

                NetworkView.ReloadContent();
                NetworkView.Desktop.ZoomToBounds();

                if (showObservers && importedProject.Observers != null)
                    RestoreObserverForms(importedProject);

                ProjectStateChanged("Project imported");
            }
            catch (Exception e)
            {
                MyLog.ERROR.WriteLine("Project import failed: " + e.Message);
            }
        }


        private bool IsProjectSaved(string fileName)
        {
            if (m_savedProjectRepresentation == null)
                return false;

            string currentRepresentation = null;

            try
            {
                currentRepresentation = GetSerializedProject(fileName);
            }
            catch
            {
                return false;
            }

            return m_savedProjectRepresentation.Equals(currentRepresentation);
        }

        private string GetSerializedProject(string fileName)
        {
            if (string.IsNullOrEmpty(fileName))
                fileName = Path.GetTempFileName();

            if (Project.Observers != null)
                MyLog.WARNING.WriteLine("Project.Observers is not null, serialization will produce side-effect!");

            Project.Observers = new List<MyAbstractObserver>();  // Potential sideffect.
            ObserverViews.ForEach(ov => { ov.StoreWindowInfo(); Project.Observers.Add(ov.Observer); });

            string originalProjectName = Project.Name;  // Avoid side-effect.
            Project.SetNameFromPath(fileName);

            string serializedProject;
            try
            {
                serializedProject = Project.Serialize(Path.GetDirectoryName(fileName));
            }
            finally
            {
                Project.Observers = null;
                Project.Name = originalProjectName;
            }

            return serializedProject;
        }
        #endregion

        #region Views

        public NodePropertyForm NodePropertyView { get; private set; }
        public MemoryBlocksForm MemoryBlocksView { get; private set; }

        public DashboardPropertyForm DashboardPropertyView { get; private set; }

        public TaskForm TaskView { get; private set; }
        public TaskPropertyForm TaskPropertyView { get; private set; }

        private GraphLayoutForm NetworkView { get; set; }
        public ConsoleForm ConsoleView { get; private set; }
        public ValidationForm ValidationView { get; private set; }
        public NodeHelpForm HelpView { get; private set; }

        public DebugForm DebugView { get; private set; }

        protected List<DockContent> m_views;
        protected ToolStripMenuItem showHideObserversMenuItem;
        public bool SuppressStateSaving { get; set; }

        public Dictionary<MyNodeGroup, GraphLayoutForm> GraphViews { get; private set; }
        public Dictionary<IScriptableNode, TextEditForm> TextEditors { get; private set; }

        public List<ObserverForm> ObserverViews { get; private set; }

        private void CreateNetworkView()
        {
            NetworkView = new GraphLayoutForm(this, Project.Network);
            NetworkView.FormClosed += GraphLayoutForm_FormClosed;

            GraphViews[Project.Network] = NetworkView;
            NetworkView.CloseButton = false;
            NetworkView.CloseButtonVisible = false;
        }

        public void CreateAndShowObserverView(MyWorkingNode node, Type observerType)
        {
            try
            {
                MyAbstractObserver observer = (MyAbstractObserver)Activator.CreateInstance(observerType);
                observer.GenericTarget = node;

                ObserverForm newView = new ObserverForm(this, observer, node);
                ObserverViews.Add(newView);

                newView.Show(dockPanel, DockState.Float);

                ProjectStateChanged("Node observer added");
            }
            catch (Exception e)
            {
                MyLog.ERROR.WriteLine("Error creating observer: " + e.Message);
            }
        }

        public void CreateAndShowObserverView(MyAbstractMemoryBlock memoryBlock, MyNode declaredOwner, Type mbObserverType)
        {
            bool isPlot = mbObserverType == typeof(MyTimePlotObserver) || mbObserverType == typeof(TimePlotObserver);

            if (isPlot && !(memoryBlock is MyMemoryBlock<float>))
            {
                MyLog.ERROR.WriteLine("Plot observers are not allowed for non-float memory blocks");
                return;
            }

            try
            {
                MyAbstractObserver observer = null;

                if (isPlot)
                {
                    if (mbObserverType == typeof(MyTimePlotObserver))
                    {
                        observer = new MyTimePlotObserver { Target = (MyMemoryBlock<float>)memoryBlock };
                    }
                    else if (mbObserverType == typeof(TimePlotObserver))
                    {
                        observer = new TimePlotObserver { Target = (MyMemoryBlock<float>)memoryBlock };
                    }
                }
                else
                {
                    MyAbstractMemoryBlockObserver memObserver = (MyAbstractMemoryBlockObserver)Activator.CreateInstance(mbObserverType);
                    memObserver.Target = memoryBlock;

                    observer = memObserver;
                }

                if (observer == null)
                    throw new InvalidOperationException("No observer was initialized");

                ObserverForm newView = new ObserverForm(this, observer, declaredOwner);
                ObserverViews.Add(newView);

                newView.Show(dockPanel, DockState.Float);

                ProjectStateChanged("Memory block observer added");
            }
            catch (Exception e)
            {
                MyLog.ERROR.WriteLine("Error creating observer: " + e.Message);
            }
        }

        public void ShowObserverView(MyAbstractObserver observer)
        {
            MyNode owner;

            var blockObserver = observer as MyAbstractMemoryBlockObserver;
            if (blockObserver != null)
            {
                owner = blockObserver.Target.Owner;
            }
            else
            {
                var plotObserver = observer as MyTimePlotObserver;
                if (plotObserver != null)
                {
                    owner = plotObserver.Target.Owner;
                }
                else
                {
                    owner = observer.GenericTarget as MyNode;
                }
            }

            var newView = new ObserverForm(this, observer, owner);
            ObserverViews.Add(newView);

            // This is only called from deserialization, the state saving would be redundant.
            //ProjectStateChanged("Abstract observer added");

            newView.Show(dockPanel, DockState.Float);
            newView.FloatPane.FloatWindow.Size = new Size((int)observer.WindowSize.Width, (int)observer.WindowSize.Height);
            newView.FloatPane.FloatWindow.Location = new Point((int)observer.WindowLocation.X, (int)observer.WindowLocation.Y);

            if (!SystemInformation.VirtualScreen.Contains(newView.FloatPane.FloatWindow.Bounds))
            {
                newView.FloatPane.FloatWindow.Location = new Point(0, 0);
            }
        }

        public void ResetObservers()
        {
            foreach (ObserverForm ov in ObserverViews.ToList())
            {
                if (ov.Observer != null)
                {
                    ov.Observer.TriggerReset();
                }
            }
        }

        public void UpdateObserverView(MyAbstractObserver observer)
        {
            foreach (ObserverForm ov in ObserverViews.ToList())
            {
                if (ov.Observer == observer)
                {
                    ov.UpdateView(SimulationHandler.SimulationStep);
                }
            }
        }

        public void UpdateObservers()
        {
            foreach (ObserverForm ov in ObserverViews.ToList())
            {
                if (ov.Observer != null)
                {
                    ov.UpdateView(SimulationHandler.SimulationStep);
                }
            }
        }

        public void CloseObservers(MyNode node)
        {
            HashSet<ObserverForm> viewsToClose = new HashSet<ObserverForm>();

            MyNodeGroup.IteratorAction checkTarget = delegate(MyNode target)
            {
                foreach (ObserverForm ov in ObserverViews)
                {
                    if (ov.Observer.GenericTarget is MyAbstractMemoryBlock)
                    {
                        if ((ov.Observer.GenericTarget as MyAbstractMemoryBlock).Owner == target)
                        {
                            viewsToClose.Add(ov);
                        }
                    }
                    else if (ov.Observer.GenericTarget == target)
                    {
                        viewsToClose.Add(ov);
                    }
                }
            };

            checkTarget(node);

            if (node is MyNodeGroup)
            {
                (node as MyNodeGroup).Iterate(true, checkTarget);
            }

            foreach (ObserverForm ov in viewsToClose)
            {
                ov.Close();
            }
        }

        public void RemoveFromDashboard(MyNode node)
        {
            DashboardPropertyView.RemovePropertiesOfNode(node);
        }

        public void RemoveObserverView(ObserverForm view)
        {
            ObserverViews.Remove(view);

            ProjectStateChanged("Abstract removed");
        }

        public TextEditForm OpenTextEditor(IScriptableNode target)
        {
            TextEditForm textEditor;

            if (!TextEditors.TryGetValue(target, out textEditor))
            {
                textEditor = new TextEditForm(this, target);
                textEditor.FormClosed += textEditor_FormClosed;
                TextEditors[target] = textEditor;
            }

            textEditor.Show(dockPanel, DockState.Document);
            return textEditor;
        }

        void textEditor_FormClosed(object sender, FormClosedEventArgs e)
        {
            TextEditors.Remove((sender as TextEditForm).Target);
        }

        internal void CloseTextEditor(IScriptableNode target)
        {
            TextEditForm textEditor;

            if (TextEditors.TryGetValue(target, out textEditor))
            {
                textEditor.Close();
            }
        }

        private void CloseAllTextEditors()
        {
            TextEditors.Values.ToList().ForEach(editor => editor.Close());
            TextEditors.Clear();
        }

        public GraphLayoutForm OpenGraphLayout(MyNodeGroup target)
        {
            GraphLayoutForm graphForm;

            if (GraphViews.ContainsKey(target))
            {
                graphForm = GraphViews[target];
            }
            else
            {
                graphForm = new GraphLayoutForm(this, target);
                graphForm.FormClosed += GraphLayoutForm_FormClosed;
                GraphViews.Add(target, graphForm);
            }

            graphForm.Show(dockPanel, DockState.Document);

            RefreshConnections(graphForm);

            SimulationHandler.Simulation.ModelChanged += graphForm.OnModelChanged;

            return graphForm;
        }

        public void RefreshConnections(GraphLayoutForm form)
        {
            SimulationHandler.RefreshTopologicalOrder();

            // Refresh the forms of the form's target parents as well.
            // The connection types might have changed for group outputs.
            var nodesToRefresh = new List<MyNode>();
            var target = form.Target;
            while (target != null)
            {
                nodesToRefresh.Add(target);
                target = target.Parent;
            }

            foreach (var graph in GraphViews)
            {
                if (nodesToRefresh.Contains(graph.Key))
                    graph.Value.RefreshGraph();
            }
        }

        internal void CloseGraphLayout(MyNodeGroup target)
        {
            if (GraphViews.ContainsKey(target))
            {
                GraphViews[target].Close();
            }
        }

        internal void ReloadGraphLayout(MyNodeGroup target)
        {
            if (GraphViews.ContainsKey(target))
            {
                GraphViews[target].ReloadContent();
            }
        }

        private void CloseAllGraphLayouts()
        {
            GraphViews.Values.ToList().ForEach(view => view.Close());
            GraphViews.Clear();
            NetworkView = null;
        }

        private void CloseAllObservers()
        {
            ObserverViews.ToList().ForEach(view => view.Close());
        }

        private void ShowHideAllObservers(bool forceShow = false)
        {
            if (forceShow && showHideObserversMenuItem.Checked)
            {
                return;
            }

            if (!showHideObserversMenuItem.Checked || forceShow)
            {
                ObserverViews.ToList().ForEach(view => view.Show());
                showHideObserversMenuItem.Checked = true;
            }
            else
            {
                ObserverViews.ToList().ForEach(view => view.Hide());
                showHideObserversMenuItem.Checked = false;
            }
        }

        private void GraphLayoutForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            var graphForm = sender as GraphLayoutForm;
            if (graphForm == null)
                return;

            SimulationHandler.Simulation.ModelChanged -= graphForm.OnModelChanged;
            GraphViews.Remove(graphForm.Target);
        }

        private bool TryRestoreViewsLayout(string layoutFileName)
        {
            if (!File.Exists(layoutFileName))
                return false;

            //TODO: change PersistString in WinFormsUI to be accessible publicly (or make our own DockContent common superclass for all forms)
            Dictionary<string, DockContent> viewTable = m_views.Where(view => !(view is GraphLayoutForm))
                .ToDictionary(view => view.GetType().ToString(), view => view);

            try
            {
                dockPanel.LoadFromXml(layoutFileName,
                    persistString => (viewTable.ContainsKey(persistString) ? viewTable[persistString] : null));
            }
            catch (Exception ex)
            {
                MyLog.WARNING.WriteLine("Unable to restore views layout (using default): " + ex.Message);

                return false;
            }

            return true;
        }

        private void ResetViewsLayout()
        {
            foreach (DockContent view in m_views)
            {
                view.Hide();
                view.DockPanel = null;
            }

            ConsoleView.Show(dockPanel, DockState.DockBottom);
            ConsoleView.DockPanel.DockBottomPortion = 0.3;

            NodePropertyView.Show(dockPanel, DockState.DockRight);
            NodePropertyView.DockPanel.DockRightPortion = 0.15;

            TaskView.Show(dockPanel, DockState.DockLeft);
            TaskView.DockPanel.DockLeftPortion = 0.15;

            MemoryBlocksView.Show(dockPanel, DockState.Float);
            MemoryBlocksView.DockHandler.FloatPane.DockTo(dockPanel, DockStyle.Right);

            TaskPropertyView.Show(dockPanel, DockState.Float);
            TaskPropertyView.DockHandler.FloatPane.DockTo(dockPanel, DockStyle.Left);

            ValidationView.Show(dockPanel, DockState.Float);
            ValidationView.DockHandler.FloatPane.DockTo(dockPanel, DockStyle.Bottom);
        }

        private void StoreViewsLayout(string layoutFileName)
        {
            dockPanel.SaveAsXml(layoutFileName);
        }

        private string UserLayoutFileName
        {
            get
            {
                return Application.LocalUserAppDataPath + "\\user.layout";
            }
        }

        #endregion

        public MainForm()
        {
            this.Font = SystemFonts.MessageBoxFont;
            InitializeComponent();


            MySimulation simulation = null;
            try
            {
                simulation = TypeMap.GetInstance<MySimulation>();
            }
            catch (Exception e)
            {
                MessageBox.Show("An error occured when initializing simulation. Please make sure you have a supported CUDA-enabled graphics card and apropriate drivers." +
                        "Technical details: " + e.Message, "Simulation Initialization Error", MessageBoxButtons.OK, MessageBoxIcon.Error);

                // this way you do not have to tweak form Close and Closing events and it works even with any worker threads still running
                Environment.Exit(1);
            }

            SimulationHandler = new MySimulationHandler(simulation);
            SimulationHandler.StateChanged += SimulationHandler_StateChanged;
            SimulationHandler.ProgressChanged += SimulationHandler_ProgressChanged;
            SimulationHandler.SimulationStopped += SimulationHandler_SimulationStopped;

            // must be created in advance to grab possible error logs
            ConsoleView = new ConsoleForm(this);

            var assemblyName = Assembly.GetExecutingAssembly().GetName();
            MyLog.INFO.WriteLine(assemblyName.Name + " version " + assemblyName.Version);


            MyConfiguration.SetupModuleSearchPath();
            MyConfiguration.ProcessCommandParams();

            try
            {
                MyConfiguration.LoadModules();
            }
            catch (Exception e)
            {
                MessageBox.Show(e.Message, "Fatal error occured during initialization", MessageBoxButtons.OK, MessageBoxIcon.Error);
                Environment.Exit(1);
            }

            UndoManager = new UndoManager(Settings.Default.UndoHistorySize);

            Documentation = new MyDocProvider();

            foreach (MyModuleConfig module in MyConfiguration.Modules)
            {
                Documentation.LoadXMLDoc(module.Assembly);
            }

            NodePropertyView = new NodePropertyForm(this);

            MemoryBlocksView = new MemoryBlocksForm(this);

            TaskView = new TaskForm(this);
            TaskPropertyView = new TaskPropertyForm(this);

            DashboardPropertyView = new DashboardPropertyForm(this);
            SimulationHandler.StateChanged += DashboardPropertyView.OnSimulationStateChanged;

            // Link the Task and Node property views to the dashboard's PropertyChanged.
            DashboardPropertyView.PropertyValueChanged += RefreshPropertyViews;

            // Link the Node and Task property views' PropertyChanged to the dashboard so that it can refresh etc.
            NodePropertyView.PropertyChanged += DashboardPropertyView.OnPropertyExternallyChanged;
            TaskPropertyView.PropertyChanged += DashboardPropertyView.OnPropertyExternallyChanged;
            TaskView.PropertyChanged += DashboardPropertyView.OnPropertyExternallyChanged;

            GraphViews = new Dictionary<MyNodeGroup, GraphLayoutForm>();
            TextEditors = new Dictionary<IScriptableNode, TextEditForm>();

            ObserverViews = new List<ObserverForm>();

            ValidationView = new ValidationForm(this, TypeMap.GetInstance<MyValidator>());
            HelpView = new NodeHelpForm(this) { StartPosition = FormStartPosition.CenterScreen };

            DebugView = new DebugForm(this);

            PopulateWorldList();
            CreateNewProject();
            CreateNetworkView();

            m_views = new List<DockContent>() { NetworkView, DashboardPropertyView, NodePropertyView, MemoryBlocksView, TaskView,
                TaskPropertyView, ConsoleView, ValidationView, DebugView, HelpView };

            foreach (var form in UIPlugins.GetBrainSimUIExtensions(this))
                m_views.Add(form);

            foreach (DockContent view in m_views)
            {
                ToolStripMenuItem viewMenuItem = new ToolStripMenuItem(view.Text);
                viewMenuItem.Click += viewToolStripMenuItem_Click;
                viewMenuItem.Tag = view;
                viewMenuItem.Name = view.Text;

                viewToolStripMenuItem.DropDownItems.Add(viewMenuItem);
            }

            ((ToolStripMenuItem)viewToolStripMenuItem.DropDownItems.Find(HelpView.Text, false).First()).ShortcutKeys = Keys.F1;
            viewToolStripMenuItem.DropDownItems.Add(new ToolStripSeparator());

            showHideObserversMenuItem = new ToolStripMenuItem("Show/Hide all observers");
            showHideObserversMenuItem.ShortcutKeys = Keys.Control | Keys.H;
            showHideObserversMenuItem.Click += showHideObserversMenuItem_Click;
            showHideObserversMenuItem.Checked = true;

            viewToolStripMenuItem.DropDownItems.Add(showHideObserversMenuItem);

            ToolStripMenuItem resetViewsMenuItem = new ToolStripMenuItem("Reset Views Layout");
            resetViewsMenuItem.ShortcutKeys = Keys.Control | Keys.W;
            resetViewsMenuItem.Click += resetViewsMenuItem_Click;

            viewToolStripMenuItem.DropDownItems.Add(resetViewsMenuItem);

            ToolStripMenuItem nodeSettingsMenuItem = new ToolStripMenuItem("Configure node selection...");
            nodeSettingsMenuItem.ShortcutKeys = Keys.Control | Keys.L;
            nodeSettingsMenuItem.Click += nodeSettingsMenuItem_Click;

            viewToolStripMenuItem.DropDownItems.Add(nodeSettingsMenuItem);

            modeDropDownList.SelectedIndex = 0;

            AddTimerMenuItem(timerToolStripSplitButton, timerItem_Click, 0);
            AddTimerMenuItem(timerToolStripSplitButton, timerItem_Click, 10);
            AddTimerMenuItem(timerToolStripSplitButton, timerItem_Click, 20);
            AddTimerMenuItem(timerToolStripSplitButton, timerItem_Click, 50);
            AddTimerMenuItem(timerToolStripSplitButton, timerItem_Click, 100);
            AddTimerMenuItem(timerToolStripSplitButton, timerItem_Click, 500);

            timerItem_Click(timerToolStripSplitButton.DropDownItems[Settings.Default.StepDelay], EventArgs.Empty);

            AddTimerMenuItem(observerTimerToolButton, observerTimerItem_Click, 0);
            AddTimerMenuItem(observerTimerToolButton, observerTimerItem_Click, 20);
            AddTimerMenuItem(observerTimerToolButton, observerTimerItem_Click, 100);
            AddTimerMenuItem(observerTimerToolButton, observerTimerItem_Click, 500);
            AddTimerMenuItem(observerTimerToolButton, observerTimerItem_Click, 1000);
            AddTimerMenuItem(observerTimerToolButton, observerTimerItem_Click, 5000);

            observerTimerItem_Click(observerTimerToolButton.DropDownItems[Settings.Default.ObserverPeriod], EventArgs.Empty);

            PropertyDescriptor descriptor = TypeDescriptor.GetProperties(typeof(MyWorkingNode))["DataFolder"];
            EditorAttribute editor = (EditorAttribute)descriptor.Attributes[typeof(EditorAttribute)];

            editor.GetType().GetField("typeName", BindingFlags.NonPublic | BindingFlags.Instance).SetValue(editor,
                typeof(MyFolderDialog).AssemblyQualifiedName);

            editor.GetType().GetField("baseTypeName", BindingFlags.NonPublic | BindingFlags.Instance).SetValue(editor,
                typeof(UITypeEditor).AssemblyQualifiedName);

            autosaveTextBox.Text = Settings.Default.AutosaveInterval.ToString();
            autosaveTextBox_Validating(this, new CancelEventArgs());

            autosaveButton.Checked = Settings.Default.AutosaveEnabled;
        }

        private void SimulationHandler_SimulationStopped(object sender, MySimulationHandler.SimulationStoppedEventArgs args)
        {
            ValidationView.UpdateListView();
            foreach (GraphLayoutForm graphView in GraphViews.Values)
                graphView.ReloadContent();
        }

        public void ProjectStateChanged(string action)
        {
            if (SuppressStateSaving)
                return;

            MyLog.DEBUG.WriteLine("State changed: {0}", action);
            SaveCurrentState(action);
            //DebugUndoManager();
        }

        private void DebugUndoManager()
        {
            MyLog.DEBUG.Write(UndoManager);
            MyLog.DEBUG.WriteLine();
        }

        private void SaveCurrentState(string action)
        {
            SaveState(GetSerializedProject(GetCurrentFileName()), GetCurrentFileName(), action);
            RefreshUndoRedoButtons();
        }

        private void SaveState(string content, string filePath, string action)
        {
            int selectedNodeId = -1;
            string selectedObserverId = null;

            var selectedNode = NodePropertyView.Target as MyNode;
            if (selectedNode != null)
                selectedNodeId = selectedNode.Id;

            var selectedObserver = NodePropertyView.Target as MyAbstractObserver;
            if (selectedObserver != null)
                selectedObserverId = selectedObserver.Id;

            var projectState = new ProjectState(content)
            {
                ProjectPath = filePath,
                Action = action,
                SelectedNode = selectedNodeId,
                SelectedObserver = selectedObserverId
            };

            projectState.GraphPanes.AddRange(GraphViews.Select(view => view.Value.Target.Id));
            var activeGraphView = GraphViews.WithIndex().FirstOrDefault(view => view.Item.Value == dockPanel.ActiveDocument);
            if (activeGraphView != null)
                projectState.SelectedGraphView = activeGraphView.Index;

            UndoManager.SaveState(projectState);
        }

        private void RefreshUndoRedoButtons()
        {
            undoButton.Enabled = SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED &&
                                 UndoManager.CanUndo();
            redoButton.Enabled = SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED &&
                                 UndoManager.CanRedo();
        }

        private void Undo()
        {
            if (SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED && UndoManager.CanUndo())
                LoadState(UndoManager.Undo());
        }

        private void Redo()
        {
            if (SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED && UndoManager.CanRedo())
                LoadState(UndoManager.Redo());
        }

        private void LoadState(ProjectState targetState)
        {
            if (targetState == null)
                return;

            LoadSerializedContent(targetState.SerializedProject, targetState.ProjectPath, Project.Name);

            // Open graph views
            foreach (MyNodeGroup nodeGroup in
                    targetState.GraphPanes.Select(nodeId => Project.GetNodeById(nodeId)).OfType<MyNodeGroup>())
                OpenGraphLayout(nodeGroup);

            // Select active graph view
            if (targetState.SelectedGraphView < dockPanel.DocumentsCount)
                (dockPanel.DocumentsToArray()[targetState.SelectedGraphView] as GraphLayoutForm).Activate();

            foreach (GraphLayoutForm graph in GraphViews.Values)
                graph.SelectNodeView(targetState.SelectedNode);

            foreach (ObserverForm observerView in ObserverViews
                .Where(observerView => observerView.Observer.Id == targetState.SelectedObserver))
                observerView.FocusWindow();

            //DebugUndoManager();
        }

        private void LoadSerializedContent(string content, string projectPath, string currentProjectName)
        {
            MyWorld oldWorld = Project == null ? null : Project.World;
            using (MyMemoryManager.Backup backup = MyMemoryManager.GetBackup())
            {
                string projectDirectory = string.IsNullOrEmpty(projectPath) ? "" : Path.GetDirectoryName(projectPath);
                Project = MyProject.Deserialize(content, projectDirectory);
                Project.Restore();
                backup.Forget();
            }

            Project.Name = currentProjectName;  // Don't use temporal dir of the project before SaveAs.

            // UI updates
            CloseCurrentProjectWindows();

            CreateNetworkView();
            OpenGraphLayout(Project.Network);

            if (Project.World != null)
                SelectWorldInWorldList(Project.World.GetType());

            RestoreObserverForms();
            RestoreDashboardForm();

            ApplyProjectOptions();

            RefreshUndoRedoButtons();

            if (WorldChanged != null)
                WorldChanged(this, new WorldChangedEventArgs(oldWorld, Project.World));
        }

        private void ApplyProjectOptions()
        {
            loadMemBlocksButton.Checked = Project.LoadAllNodesData;
            saveMemBlocksButton.Checked = Project.SaveAllNodesData;
        }

        private void RestoreObserverForms(MyProject project = null)
        {
            if (project == null)
                project = Project;

            foreach (MyAbstractObserver observer in project.Observers.Where(observer => observer.GenericTarget != null))
                ShowObserverView(observer);

            project.Observers = null;
        }

        private void RestoreDashboardForm()
        {
            DashboardPropertyView.SetDashboards(Project.Dashboard, Project.GroupedDashboard);
        }

        private string GetCurrentFileName()
        {
            return !string.IsNullOrEmpty(saveFileDialog.FileName)
                ? MyProject.MakeNameFromPath(saveFileDialog.FileName)
                : "";
        }

        private void RefreshPropertyViews(object s, PropertyValueChangedEventArgs e)
        {
            NodePropertyView.RefreshView();
            TaskPropertyView.RefreshView();
            TaskView.RefreshView();
        }

        public void PopulateWorldList()
        {
            int itemIndex = 0;

            foreach (MyWorldConfig wc in MyConfiguration.KnownWorlds.Values)
            {
                bool isAmongToolBarNodes = (Settings.Default.ToolBarNodes != null &&
                    Settings.Default.ToolBarNodes.Contains(wc.NodeType.Name));
                if (isAmongToolBarNodes || wc.IsBasicNode)
                {
                    if (!worldList.Items.Contains(wc))
                        worldList.Items.Insert(itemIndex, wc);
                    itemIndex++;
                }
                else
                {
                    if (worldList.Items.Contains(wc))
                        worldList.Items.RemoveAt(itemIndex);
                }
            }
        }

        public void SelectWorldInWorldList(Type worldType)
        {
            if (Settings.Default.ToolBarNodes == null)
            {
                AppSettings.SaveSettings(settings => settings.ToolBarNodes = new System.Collections.Specialized.StringCollection());
            }

            // if the world is not present in the combo box, add it first
            if (!Settings.Default.ToolBarNodes.Contains(worldType.Name))
            {
                Settings.Default.ToolBarNodes.Add(worldType.Name);
                worldList.Items.Add(MyConfiguration.KnownWorlds[worldType]);
            }

            worldList.SelectedItem = MyConfiguration.KnownWorlds[worldType];
        }

        private void AddTimerMenuItem(ToolStripSplitButton splitButton, EventHandler clickHandler, int ms)
        {
            string title;

            if (ms < 1000)
            {
                title = ms + " ms";
            }
            else
            {
                title = (ms / 1000.0f).ToString("0.##" + " s");
            }

            ToolStripMenuItem mi = new ToolStripMenuItem(title);
            mi.Tag = ms;
            mi.Click += clickHandler;

            splitButton.DropDownItems.Add(mi);
        }

        void resetViewsMenuItem_Click(object sender, EventArgs e)
        {
            ResetViewsLayout();
        }

        void showHideObserversMenuItem_Click(object sender, EventArgs e)
        {
            ShowHideAllObservers();
        }

        void timerItem_Click(object sender, EventArgs e)
        {
            for (int i = 0; i < timerToolStripSplitButton.DropDownItems.Count; i++)
            {
                ToolStripMenuItem item = (ToolStripMenuItem)timerToolStripSplitButton.DropDownItems[i];

                if (item == sender)
                {
                    AppSettings.SaveSettings(settings => settings.StepDelay = i);
                    SimulationHandler.SleepInterval = (int)item.Tag;
                    item.Checked = true;
                }
                else
                {
                    item.Checked = false;
                }
            }
        }

        void observerTimerItem_Click(object sender, EventArgs e)
        {
            for (int i = 0; i < observerTimerToolButton.DropDownItems.Count; i++)
            {
                ToolStripMenuItem item = (ToolStripMenuItem)observerTimerToolButton.DropDownItems[i];

                if (item == sender)
                {
                    AppSettings.SaveSettings(settings => settings.ObserverPeriod = i);
                    SimulationHandler.ReportInterval = (int)item.Tag;
                    item.Checked = true;
                }
                else
                {
                    item.Checked = false;
                }
            }
        }

        #region Simulation

        /// <summary>
        /// Shorthand for StartSimulation(stepCount: 1)
        /// </summary>
        private void StartSimulationStep()
        {
            StartSimulation(stepCount: 1);
        }

        // TODO: move all bits related to simulation out of here and into SimulationHandler, leave only the GUI-specific things.
        /// <summary>
        /// Starts simulation.
        /// </summary>
        /// <param name="stepCount">Number of simulation steps to run, 0 means unlimited.</param>
        private void StartSimulation(uint stepCount = 0)
        {
            if (SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED)
            {
                MyLog.INFO.WriteLine("--------------");
                bool anyOutputChanged = false;
                try
                {
                    // TODO(HonzaS): Move this into Simulation, it will be reused when the model is changed on the run.
                    anyOutputChanged = SimulationHandler.UpdateMemoryModel();
                }
                finally
                {
                    // Error handling is done below in a validation assert.
                }

                // Perform project validation (world and network).
                SimulationHandler.Simulation.Validate(Project);

                MyValidator validator = SimulationHandler.Simulation.Validator;

                if (ObserverViews != null)
                    ObserverViews.ForEach(view => view.Observer.Validate(validator));

                validator.AssertError(!anyOutputChanged, Project.Network, "Possible infinite loop in memory block sizes.");

                ValidationView.UpdateListView();

                ResetObservers();

                if (validator.ValidationSucessfull)
                {
                    try
                    {
                        SimulationHandler.StartSimulation(stepCount);
                    }
                    catch (Exception e)
                    {
                        MyLog.ERROR.WriteLine("Simulation cannot be started! Exception occured: " + e.Message);
                    }
                }
                else
                {
                    MyLog.ERROR.WriteLine("Simulation cannot be started! Validation failed.");
                    OpenFloatingOrActivate(ValidationView);
                }
            }
            else
            {
                try
                {
                    SimulationHandler.StartSimulation(stepCount);
                }
                catch (Exception e)
                {
                    MyLog.ERROR.WriteLine("Simulation cannot be started! Exception occured: " + e.Message);
                }
            }
        }

        void SimulationHandler_ProgressChanged(object sender, System.ComponentModel.ProgressChangedEventArgs e)
        {
            if (SimulationHandler.State != MySimulationHandler.SimulationState.STOPPED)
            {
                statusStrip.BeginInvoke((MethodInvoker)(() => stepStatusLabel.Text = "(" + SimulationHandler.SimulationStep + ", " + SimulationHandler.SimulationSpeed + "/s)"));

                if (dockPanel.ActiveDocument is GraphLayoutForm)
                {
                    GraphLayoutForm activeLayout = dockPanel.ActiveDocument as GraphLayoutForm;
                    activeLayout.Desktop.Invalidate();
                }
            }
            else
            {
                statusStrip.Invoke((MethodInvoker)(() => stepStatusLabel.Text = String.Empty));
            }
        }

        /// <summary>
        /// This method is mainly for extensive UI action (e.g. modal dialogs) to be executed during simulation
        /// </summary>
        /// <param name="actionToExecute">Action should return true if resume of simulation is needed after the action is finished</param>
        internal void PauseSimulationForAction(Func<bool> actionToExecute)
        {
            bool wasRunning = false;

            if (SimulationHandler.State == MySimulationHandler.SimulationState.RUNNING)
            {
                SimulationHandler.PauseSimulation();
                wasRunning = true;
            }

            bool resume = actionToExecute();

            if (wasRunning && resume)
            {
                SimulationHandler.StartSimulation();
            }
        }

        #endregion

        #region Global Shortcuts

        public bool PerformMainMenuClick(Keys shortCut)
        {
            foreach (ToolStripItem item in mainMenuStrip.Items)
            {
                if (PerformMenuClick(item, shortCut))
                {
                    return true;
                }
            }
            return false;
        }

        private bool PerformMenuClick(ToolStripItem menuItem, Keys shortCut)
        {
            if (menuItem is ToolStripMenuItem)
            {
                if ((menuItem as ToolStripMenuItem).ShortcutKeys == shortCut)
                {
                    menuItem.PerformClick();
                    return true;
                }
            }

            if (menuItem is ToolStripDropDownItem)
            {
                foreach (ToolStripItem item in (menuItem as ToolStripDropDownItem).DropDownItems)
                {
                    if (PerformMenuClick(item, shortCut))
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        #endregion

        #region Clipboard Copy/Paste

        public void CopySelectedNodesToClipboard()
        {
            if (dockPanel.ActiveDocument is GraphLayoutForm)
            {
                GraphLayoutForm activeLayout = dockPanel.ActiveDocument as GraphLayoutForm;

                NodeSelection selection = null;

                if (activeLayout.Desktop.FocusElement is NodeSelection)
                {
                    selection = activeLayout.Desktop.FocusElement as NodeSelection;
                }
                else if (activeLayout.Desktop.FocusElement is Node)
                {
                    selection = new NodeSelection(new Node[] { activeLayout.Desktop.FocusElement as Node });
                }

                if (selection != null)
                {
                    HashSet<int> approvedNodes = new HashSet<int>();

                    var clipboardNetwork = Project.CreateNode<MyNetwork>();
                    clipboardNetwork.Name = "Clipboard";

                    foreach (MyNodeView nodeView in selection.Nodes)
                    {
                        MyNode selectedNode = nodeView.Node;

                        if (selectedNode is MyWorkingNode)
                        {
                            clipboardNetwork.Children.Add(nodeView.Node);
                            approvedNodes.Add(selectedNode.Id);
                        }

                        if (selectedNode is MyNodeGroup)
                        {
                            (selectedNode as MyNodeGroup).Iterate(true, true, node => approvedNodes.Add(node.Id));
                        }
                    }

                    if (approvedNodes.Count > 0)
                    {
                        clipboardNetwork.PrepareConnections();
                        clipboardNetwork.FilterPreparedCollection(approvedNodes);

                        YAXSerializer networkSerializer = new YAXSerializer(typeof(MyNetwork), YAXExceptionHandlingPolicies.ThrowErrorsOnly, YAXExceptionTypes.Warning, YAXSerializationOptions.DontSerializeNullObjects);
                        string xml = networkSerializer.Serialize(clipboardNetwork);

                        Clipboard.SetText(xml);
                    }
                    else
                    {
                        MyLog.WARNING.WriteLine("Copying is not allowed");
                    }
                }
                else
                {
                    MyLog.WARNING.WriteLine("Selection is empty");
                }
            }
        }

        public void PasteNodesFromClipboard()
        {
            if (Clipboard.ContainsText() && dockPanel.ActiveDocument is GraphLayoutForm)
            {
                string xml = Clipboard.GetText();

                try
                {
                    YAXSerializer networkSerializer = new YAXSerializer(typeof(MyNetwork), YAXExceptionHandlingPolicies.ThrowErrorsOnly, YAXExceptionTypes.Error, YAXSerializationOptions.DontSerializeNullObjects);

                    MyNetwork networkToPaste = (MyNetwork)networkSerializer.Deserialize(xml);
                    networkToPaste.UpdateAfterDeserialization(0, Project, showWarnings: false);

                    GraphLayoutForm activeLayout = dockPanel.ActiveDocument as GraphLayoutForm;

                    activeLayout.Target.AppendGroupContent(networkToPaste);
                    activeLayout.ReloadContent();

                    HashSet<int> pastedNodes = new HashSet<int>();
                    networkToPaste.Children.ForEach(node => pastedNodes.Add(node.Id));

                    List<MyNodeView> pastedNodeViews = new List<MyNodeView>();
                    RectangleF? pastedBounds = null;
                    Graphics context = activeLayout.Desktop.CreateGraphics();

                    foreach (MyNodeView nodeView in activeLayout.Desktop.Nodes)
                    {
                        if (pastedNodes.Contains(nodeView.Node.Id))
                        {
                            pastedNodeViews.Add(nodeView);

                            SizeF size = GraphRenderer.Measure(context, nodeView);
                            RectangleF bounds = new RectangleF(nodeView.Location, size);

                            if (pastedBounds.HasValue)
                            {
                                pastedBounds = RectangleF.Union(pastedBounds.Value, bounds);
                            }
                            else
                            {
                                pastedBounds = bounds;
                            }
                        }
                    }

                    PointF pasteLocation = activeLayout.Desktop.UnprojectPoint(new PointF(20, 20));

                    if (pastedBounds.HasValue)
                    {
                        foreach (MyNodeView nodeView in pastedNodeViews)
                        {
                            nodeView.Node.Location = new MyLocation()
                            {
                                X = nodeView.Location.X - pastedBounds.Value.Left + pasteLocation.X,
                                Y = nodeView.Location.Y - pastedBounds.Value.Top + pasteLocation.Y
                            };
                            nodeView.UpdateView();
                        }
                    }


                    //select pasted nodes
                    activeLayout.Desktop.FocusElement = new NodeSelection(pastedNodeViews);

                    ProjectStateChanged("Nodes pasted from clipboard");
                }
                catch (Exception e)
                {
                    MyLog.ERROR.WriteLine("Paste failed: " + e.Message);
                }
            }
        }

        #endregion

        public bool CheckDashboardContains(object target, string propertyName)
        {
            if (Project.Dashboard == null)
                return false;
            return Project.Dashboard.Contains(target, propertyName);
        }

        public void DashboardPropertyToggle(object target, string propertyName, bool active)
        {
            string action = null;
            if (active)
            {
                if (Project.Dashboard.Add(target, propertyName))
                    action = "Dashboard property added";
            }
            else
            {
                if (Project.Dashboard.Remove(target, propertyName))
                    action = "Dashboard property removed";
            }

            if (action != null)
                ProjectStateChanged(action + ": " + propertyName);
        }

        public void InvalidateGraphLayouts()
        {
            foreach (GraphLayoutForm graphLayoutForm in GraphViews.Values)
                graphLayoutForm.Desktop.Invalidate();
        }
    }
}
