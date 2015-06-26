using BrainSimulator.Configuration;
using BrainSimulator.Execution;
using BrainSimulator.Memory;
using BrainSimulator.Nodes;
using BrainSimulator.Observers;
using BrainSimulator.Utils;
using BrainSimulatorGUI.Nodes;
using BrainSimulatorGUI.NodeView;
using BrainSimulatorGUI.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace BrainSimulatorGUI.Forms
{
    public partial class NodePropertyForm : DockContent
    {
        private MainForm m_mainForm;

        public bool CanEdit
        {
            set
            {                
                propertyGrid.Enabled = value || Target is MyAbstractObserver;
            }
        }

        public object Target
        {
            get { return propertyGrid.SelectedObject; }
            set { 
                propertyGrid.SelectedObject = value;

                UpdateTitleAndButtons();                    
                UpdateObserverList();                
            }
        }

        public NodePropertyForm(MainForm mainForm)
        {
            m_mainForm = mainForm;
            InitializeComponent();                   
            propertyGrid.BrowsableAttributes = new AttributeCollection(new MyBrowsableAttribute());            
        }

        private void propertyGrid_PropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            MyNodeView nodeView = null;

            foreach(GraphLayoutForm graphView in m_mainForm.GraphViews.Values) {

                if (graphView.Desktop.FocusElement is MyNodeView 
                    && (graphView.Desktop.FocusElement as MyNodeView).Node == propertyGrid.SelectedObject)
                {
                    nodeView = graphView.Desktop.FocusElement as MyNodeView;                    
                    nodeView.UpdateView();               
                }

                if (propertyGrid.SelectedObject is MyNodeGroup && graphView.Target == propertyGrid.SelectedObject)
                {
                    graphView.Text = graphView.Target.Name;
                }

                graphView.Desktop.Invalidate();
            }

            if (nodeView != null)
            {
                if (nodeView.BranchChangeNeeded)
                {
                    if (nodeView.OutputBranchChangeNeeded)
                    {
                        m_mainForm.CloseObservers(nodeView.Node);
                    }

                    if (nodeView.Node is MyNodeGroup)
                    {
                        m_mainForm.ReloadGraphLayout(nodeView.Node as MyNodeGroup);
                    }

                    (nodeView as MyVariableBranchView).UpdateBranches();
                }
            }            

            propertyGrid.Refresh();

            if (Target is MyNode) 
            {
                UpdateTitleAndButtons();            
                m_mainForm.MemoryBlocksView.UpdateView();
            }

            if (Target is MyAbstractObserver && m_mainForm.SimulationHandler.State == MySimulationHandler.SimulationState.PAUSED)
            {
                m_mainForm.UpdateObserverView(Target as MyAbstractObserver);
            }
        }

        private void UpdateTitleAndButtons()
        {
            if (Target is MyNode)
            {
                MyNode node = Target as MyNode;
                nodeNameTextBox.Rtf = @"{\rtf1\ansi \b " + node.Name + @"\b0  - " + node.GetType().Name + "}";
            }
            else if (Target is MyAbstractObserver)
            {
                MyAbstractObserver observer = Target as MyAbstractObserver;
                nodeNameTextBox.Rtf = @"{\rtf1\ansi \b " + observer.TargetIdentifier + @"\b0  - " + observer.GetType().Name + "}";

                snapshotButton.Checked = observer.AutosaveSnapshop;
            }
            else
            {
                nodeNameTextBox.Rtf = "";
            }

            CanEdit = m_mainForm.SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED;
            helpButton.Enabled = Target is MyWorkingNode || Target is MyAbstractObserver;

            snapshotButton.Enabled = Target is MyAbstractObserver;

            if (Target is MyWorkingNode)
            {
                saveNodeDataButton.Enabled = true;
                saveNodeDataButton.Checked = (Target as MyWorkingNode).SaveOnStop;

                loadNodeDataButton.Enabled = true;
                loadNodeDataButton.Checked = (Target as MyWorkingNode).LoadOnStart;

                clearDataButton.Enabled = true;
            }
            else
            {
                saveNodeDataButton.Enabled = false;
                saveNodeDataButton.Checked = false;

                loadNodeDataButton.Enabled = false;
                loadNodeDataButton.Checked = false;

                clearDataButton.Enabled = false;
            }
        }

        private void UpdateObserverList()
        {
            observerDropDownButton.DropDownItems.Clear();
            observerDropDownButton.Enabled = false;

            Dictionary<Type, MyObserverConfig> observers = null;

            if (Target is MyWorkingNode && MyConfiguration.KnownNodes.ContainsKey(Target.GetType()))
            {
                observers = MyConfiguration.KnownNodes[Target.GetType()].KnownObservers;
            }
            else if (Target is MyWorld && MyConfiguration.KnownWorlds.ContainsKey(Target.GetType())) 
            {
                observers = MyConfiguration.KnownWorlds[Target.GetType()].KnownObservers;
            }

            if (observers != null && observers.Count > 0)
            {
                foreach (MyObserverConfig oc in observers.Values)
                {                        
                    ToolStripMenuItem item = new ToolStripMenuItem(oc.ObserverType.Name.Substring(2));
                    item.Tag = oc.ObserverType;
                    item.Click += item_Click;
                    observerDropDownButton.DropDownItems.Add(item);
                }
                observerDropDownButton.Enabled = true;
            }            
        }

        void item_Click(object sender, EventArgs e)
        {
            m_mainForm.CreateAndShowObserverView(Target as MyWorkingNode, (sender as ToolStripMenuItem).Tag as Type);
        }

        private void saveDataNodeButton_Click(object sender, EventArgs e)
        {
            /*
            if (folderBrowserDialog.ShowDialog(this) == DialogResult.OK)
            {
                MyMemoryManager.Instance.LoadBlocks(Target as MyNode, true, folderBrowserDialog.SelectedPath, false);
                MyLog.INFO.WriteLine("Memory blocks of '" + (Target as MyNode).Name + "' were loaded from '" + folderBrowserDialog.SelectedPath + "'.");
            }       
            */

            if (Target is MyWorkingNode) 
            {
                (Target as MyWorkingNode).SaveOnStop = saveNodeDataButton.Checked;
                propertyGrid.Refresh();
            }
        }

        private void loadNodeDataButton_Click(object sender, EventArgs e)
        {
            if (Target is MyWorkingNode)
            {
                (Target as MyWorkingNode).LoadOnStart = loadNodeDataButton.Checked;
                propertyGrid.Refresh();
            }
        }

        private void helpButton_Click(object sender, EventArgs e)
        {
            m_mainForm.OpenNodeHelpView();
        }

        private void clearDataButton_Click(object sender, EventArgs e)
        {
            if (MessageBox.Show("Clear all temporal data for node: " + (Target as MyWorkingNode).Name + "?",
                "Clear data", MessageBoxButtons.OKCancel, MessageBoxIcon.Question) == DialogResult.OK)
            {
                MyMemoryBlockSerializer.ClearTempData(Target as MyWorkingNode);               
            }
        }

        private void snapshotButton_Click(object sender, EventArgs e)
        {
            (Target as MyAbstractObserver).AutosaveSnapshop = snapshotButton.Checked;
        }
    }
}
