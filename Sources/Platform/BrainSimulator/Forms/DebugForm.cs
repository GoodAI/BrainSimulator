using Aga.Controls.Tree;
using Aga.Controls.Tree.NodeControls;
using BrainSimulator;
using BrainSimulator.Execution;
using BrainSimulator.Nodes;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace BrainSimulatorGUI.Forms
{
    public partial class DebugForm : DockContent
    {
        private MainForm m_mainForm;        
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

            if (executable is MyExecutionBlock)
            {
                foreach (IMyExecutable child in (executable as MyExecutionBlock).Children)
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

            return result;
        }

        private void UpdateDebugListView()
        {
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
            MySimulationHandler simulatinHandler = sender as MySimulationHandler;

            runToolButton.Enabled = simulatinHandler.CanStart;
            stepInButton.Enabled = simulatinHandler.CanStepInto;
            stepOutButton.Enabled = simulatinHandler.CanStepOut;
            stepOverButton.Enabled = simulatinHandler.CanStepOver;
            pauseToolButton.Enabled = simulatinHandler.CanPause;

            if (e.NewState == MySimulationHandler.SimulationState.PAUSED && simulatinHandler.Simulation.InDebugMode)
            {
                noDebugLabel.Visible = false;
                toolStrip.Enabled = true;                                               

                if (m_executionPlan == null)
                {
                    UpdateDebugListView();
                }

                MyExecutionBlock currentBlock = simulatinHandler.Simulation.CurrentDebuggedBlocks[0];
                m_selectedNodeView = null;

                if (currentBlock != null && currentBlock.CurrentChild != null)
                {                
                    m_selectedNodeView = debugTreeView.AllNodes.FirstOrDefault(node => (node.Tag is MyDebugNode && (node.Tag as MyDebugNode).Executable == currentBlock.CurrentChild));
                    
                };

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
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            return m_mainForm.PerformMainMenuClick(keyData);
        }

        private void nodeTextBox1_DrawText(object sender, DrawEventArgs e)
        {
            AlterText(e);
            AlterBackgroud(e);         
        }

        private void nodeTextBox2_DrawText(object sender, DrawEventArgs e)
        {
            AlterText(e);
            AlterBackgroud(e);
        }

        private void AlterBackgroud(DrawEventArgs e)
        {
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
    }

    public class MyDebugNode : Node
    {
        //public virtual bool Checked { get { return false; } }
        public virtual Image Icon { get; protected set; }
        public string OwnerName { get; protected set; }

        public IMyExecutable Executable { get; private set; }

        public MyDebugNode(IMyExecutable executable): base(executable.Name)
        {
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
