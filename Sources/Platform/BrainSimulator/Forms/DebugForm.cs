using Aga.Controls.Tree;
using Aga.Controls.Tree.NodeControls;
using GoodAI.Core.Execution;
using GoodAI.Core.Task;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.BrainSimulator.Forms
{
    public partial class DebugForm : DockContent
    {
        private readonly MainForm m_mainForm;        
        private MyExecutionPlan[] m_executionPlan;


        private MyDebugNode CreateDebugNode(IMyExecutable executable)
        {
            MyDebugNode result;

            if (executable is MyTask)
            {
                result = new MyDebugTaskNode(executable as MyTask);
            }
            else
            {
                result = new MyDebugNode(executable);
            }

            var executableBlock = executable as MyExecutionBlock;
            if (executableBlock != null)
            {
                foreach (IMyExecutable child in executableBlock.Children)
                {
                    if (child is MySignalTask)
                    {
                        if (showSignalsButton.Checked)
                        {
                            result.Nodes.Add(CreateDebugNode(child));
                        }
                    }
                    else if (showDisabledTasksButton.Checked || child.Enabled)
                    {
                        result.Nodes.Add(CreateDebugNode(child));
                    }
                }
            }

            result.BreakpointStateChanged += OnBreakpointStateChanged;

            return result;
        }

        private void OnBreakpointStateChanged(object sender, MyDebugNode.BreakpointEventArgs args)
        {
            if (args.Node.Breakpoint)
                m_mainForm.Breakpoints.Add(args.Node.Executable);
            else
                m_mainForm.Breakpoints.Remove(args.Node.Executable);
        }

        private void UpdateDebugListView()
        {
            // Clean up the event handlers.
            foreach (var node in debugTreeView.AllNodes)
            {
                var debugNode = node.Tag as MyDebugNode;
                debugNode.BreakpointStateChanged -= OnBreakpointStateChanged;
            }

            m_executionPlan = m_mainForm.SimulationHandler.Simulation.ExecutionPlan;

            if (m_executionPlan != null)
            {
                TreeModel treeModel = new TreeModel();

                for (int i = 0; i < 1; i++)
                {
                    treeModel.Nodes.Add(CreateDebugNode(m_executionPlan[i].InitStepPlan));
                    treeModel.Nodes.Add(CreateDebugNode(m_executionPlan[i].StandardStepPlan));
                }

                debugTreeView.Model = treeModel;
                debugTreeView.ExpandAll();
            }

            foreach (var node in debugTreeView.AllNodes)
            {
                var debugNode = node.Tag as MyDebugNode;
                debugNode.Breakpoint = m_mainForm.Breakpoints.Contains(debugNode.Executable);
            }
        }

        public DebugForm(MainForm mainForm)
        {
            m_mainForm = mainForm;
            InitializeComponent();

            m_mainForm.SimulationHandler.StateChanged += SimulationHandler_StateChanged;
        }

        TreeNodeAdv m_selectedNodeView = null;

        void SimulationHandler_StateChanged(object sender, MySimulationHandler.StateEventArgs e)
        {
            MySimulationHandler simulationHandler = sender as MySimulationHandler;

            runToolButton.Enabled = simulationHandler.CanStart;
            stepInButton.Enabled = simulationHandler.CanStepInto;
            stepOutButton.Enabled = simulationHandler.CanStepOut;
            stepOverButton.Enabled = simulationHandler.CanStepOver;
            pauseToolButton.Enabled = simulationHandler.CanPause;

            if (e.NewState == MySimulationHandler.SimulationState.RUNNING)
            {
                if (m_executionPlan == null)
                    UpdateDebugListView();
            }
            else if (e.NewState == MySimulationHandler.SimulationState.PAUSED)
            {
                if (m_executionPlan == null)
                    UpdateDebugListView();

                if (simulationHandler.Simulation.InDebugMode)
                {
                    toolStrip.Enabled = true;                                               
                    noDebugLabel.Visible = false;

                    MyExecutionBlock currentBlock = simulationHandler.Simulation.CurrentDebuggedBlocks[0];
                    m_selectedNodeView = null;

                    if (currentBlock != null && currentBlock.CurrentChild != null)
                    {                
                        m_selectedNodeView = debugTreeView.AllNodes.FirstOrDefault(node => (node.Tag is MyDebugNode && (node.Tag as MyDebugNode).Executable == currentBlock.CurrentChild));
                        
                    };
                }

                debugTreeView.Invalidate();                
                //debugTreeView.Invoke((MethodInvoker)(() => debugTreeView.SelectedNode = m_selectedNodeView));
            }
            else if (e.NewState == MySimulationHandler.SimulationState.STOPPED)
            {
                m_executionPlan = null;
                debugTreeView.Model = null;
                noDebugLabel.Visible = true;
                toolStrip.Enabled = false;

                if (this.IsFloat)
                {
                    this.Hide();
                }
            }

            breakpointCheckBox.EditEnabled = simulationHandler.Simulation.InDebugMode;
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            return m_mainForm.PerformMainMenuClick(keyData);
        }

        private void nodeTextBox1_DrawText(object sender, DrawEventArgs e)
        {
            AlterText(e);
            AlterBackground(e);         
        }

        private void profilerTimeValue_DrawText(object sender, DrawEventArgs e)
        {
            var parentTreeNode = e.Node.Parent;
            var parentDebugNode = parentTreeNode.Tag as MyDebugNode;
            if (parentDebugNode != null)
            {
                var parentBlock = parentDebugNode.Executable as MyExecutionBlock;
                if (parentBlock != null)
                {
                    // The parent node contains an executable block.
                    var debugNode = e.Node.Tag as MyDebugNode;

                    // Fill the time property in the view model.
                    TimeSpan profilingTime;
                    if (parentBlock.ProfilingInfo.TryGetValue(debugNode.Executable, out profilingTime))
                        debugNode.ProfilerTime = profilingTime;

                    TreeNodeAdv selectedTreeNode = GetSelectedTreeNode();
                    if (selectedTreeNode == null)
                        return;

                    // If the this node should be colored according to the time it took, use the pre-calculated color
                    // in the draw event.
                    if (parentTreeNode == selectedTreeNode)
                        e.BackgroundBrush = new SolidBrush(debugNode.BackgroundColor);
                }
            }
        }

        private void AlterBackground(DrawEventArgs e)
        {
            var nodeData = e.Node.Tag as MyDebugNode;
            if (nodeData != null && nodeData.Breakpoint && e.Node != debugTreeView.SelectedNode)
            {
                e.BackgroundBrush = Brushes.IndianRed;
            }

            if (e.Node == m_selectedNodeView && m_selectedNodeView != debugTreeView.SelectedNode)
            {
                e.BackgroundBrush = Brushes.Gold;
            }
        }

        private static void AlterText(DrawEventArgs e)
        {
            if (e.Node.Tag is MyDebugNode)
            {
                IMyExecutable executable = (e.Node.Tag as MyDebugNode).Executable;
                if (executable is MyExecutionBlock)
                {
                    e.Font = new Font(e.Font, FontStyle.Bold);
                }

                if (!executable.Enabled)
                {
                    e.TextColor = SystemColors.GrayText;
                }
            }
        }

        private void nodeCheckBox1_IsVisibleValueNeeded(object sender, Aga.Controls.Tree.NodeControls.NodeControlValueEventArgs e)
        {
            e.Value = e.Node.Tag is MyDebugTaskNode;
        }

        private void showDisabledTasksButton_CheckedChanged(object sender, EventArgs e)
        {
            UpdateDebugListView();
        }

        private void showSignalsButton_CheckedChanged(object sender, EventArgs e)
        {
            UpdateDebugListView();
        }

        private void runToolButton_Click(object sender, EventArgs e)
        {
            m_mainForm.runToolButton.PerformClick();
        }

        private void pauseToolButton_Click(object sender, EventArgs e)
        {
            m_mainForm.pauseToolButton.PerformClick();
        }

        private void stopToolButton_Click(object sender, EventArgs e)
        {
            m_mainForm.stopToolButton.PerformClick();
        }

        private void stepOverButton_Click(object sender, EventArgs e)
        {
            m_mainForm.stepOverToolStripMenuItem.PerformClick();
        }

        private void stepInButton_Click(object sender, EventArgs e)
        {
            m_mainForm.stepIntoToolStripMenuItem.PerformClick();
        }

        private void stepOutButton_Click(object sender, EventArgs e)
        {
            m_mainForm.stepOutToolStripMenuItem.PerformClick();
        }

        private void debugTreeView_SelectionChanged(object sender, EventArgs e)
        {
            // Color the value according to profiling times.
            if (!MyExecutionBlock.IsProfiling)
                return;

            TreeNodeAdv selectedTreeNode = GetSelectedTreeNode();
            if (selectedTreeNode == null)
                return;

            // Get the relevant children of the current node.
            var children = selectedTreeNode.Children
                .Select(child => child.Tag as MyDebugNode)
                .Where(childDebugNode => childDebugNode != null && childDebugNode.ProfilerTime != null)
                .ToList();

            // Calculate total time of the individual components.
            double totalTime = children.Sum(childDebugNode => childDebugNode.ProfilerTime.Value.TotalMilliseconds);

            // Calculate the colors of the children nodes.
            foreach (MyDebugNode debugNodeChild in children)
            {
                var saturation = (int) (255 * debugNodeChild.ProfilerTime.Value.TotalMilliseconds/totalTime);
                // ~ 1% filter.
                if (saturation > 3)
                    debugNodeChild.BackgroundColor = Color.FromArgb(saturation, 255, 0, 0);
            }
        }

        private TreeNodeAdv GetSelectedTreeNode()
        {
            TreeNodeAdv selectedTreeNode = debugTreeView.SelectedNode;

            if (selectedTreeNode == null)
                selectedTreeNode = debugTreeView.AllNodes.FirstOrDefault(node => node.ToString().StartsWith("Simulation"));
            return selectedTreeNode;
        }
    }

    public class MyDebugNode : Node, IDisposable
    {
        public class BreakpointEventArgs : EventArgs
        {
            public MyDebugNode Node { get; set; }

            public BreakpointEventArgs(MyDebugNode node)
            {
                Node = node;
            }
        }

        public delegate void BreakpointStateChangedEvent(object sender, BreakpointEventArgs args);

        public event BreakpointStateChangedEvent BreakpointStateChanged;
        //public virtual bool Checked { get { return false; } }
        public virtual Image Icon { get; protected set; }
        public string OwnerName { get; protected set; }

        private bool m_breakpoint;
        public bool Breakpoint
        {
            get
            {
                return m_breakpoint;
            }
            set
            {
                m_breakpoint = value;
                if (BreakpointStateChanged != null)
                    BreakpointStateChanged(this, new BreakpointEventArgs(this));
            }
        }

        public TimeSpan? ProfilerTime { get; set; }

        public string ProfilerTimeFormatted
        {
            get
            {
                if (ProfilerTime.HasValue)
                    return string.Format("{0}ms", ProfilerTime.Value.TotalMilliseconds);
                else
                    return string.Empty;
            }
        }

        public Color BackgroundColor { get; set; }

        public IMyExecutable Executable { get; private set; }

        public MyDebugNode(IMyExecutable executable): base(executable.Name)
        {
            BackgroundColor = Color.White;

            Executable = executable;

            if (Executable is MyIncomingSignalTask)
            {
                Icon = Properties.Resources.signal_in;
            }
            else if (Executable is MyOutgoingSignalTask)
            {
                Icon = Properties.Resources.signal_out;
            }
            else
            {
                Icon = Properties.Resources.gear_16xLG;                
            }
        }

        public void Dispose()
        {
            BreakpointStateChanged = null;
        }
    }

    public class MyDebugTaskNode : MyDebugNode
    {
        public bool Checked
        {
            get
            {
                return Executable.Enabled;
            }
        }

        public MyDebugTaskNode(MyTask task): base(task)
        {
            Icon = Properties.Resources.gears;
            OwnerName = task.GetType().Name;
        }
    }
}
