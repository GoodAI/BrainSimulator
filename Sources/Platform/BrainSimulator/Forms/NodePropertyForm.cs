using GoodAI.BrainSimulator.NodeView;
using GoodAI.Core.Configuration;
using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Forms;
using GoodAI.Core.Dashboard;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.BrainSimulator.Forms
{
    public partial class NodePropertyForm : DockContent, INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;

        protected virtual void OnPropertyChanged(string propertyName = null)
        {
            if (PropertyChanged != null)
                PropertyChanged(this, new PropertyChangedEventArgs(propertyName));

            m_mainForm.ProjectStateChanged(string.Format("Node property value changed: {0}", propertyName));
        }

        private readonly MainForm m_mainForm;

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

                if (!(value is MyNode))
                {
                    dashboardButton.Enabled = false;
                }

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
            OnPropertyChanged(e.ChangedItem.PropertyDescriptor.Name);

            var node = propertyGrid.SelectedObject as MyNode;
            if (node != null)
            {
                var nodeGroup = node as MyNodeGroup;
                if (nodeGroup != null)
                    m_mainForm.ReloadGraphLayout(nodeGroup);

                node.Updated();
            }

            foreach (GraphLayoutForm graphView in m_mainForm.GraphViews.Values)
            {
                if (propertyGrid.SelectedObject is MyNodeGroup && graphView.Target == propertyGrid.SelectedObject)
                    graphView.Text = graphView.Target.Name;
            }

            foreach (TextEditForm textEditor in m_mainForm.TextEditors.Values)
            {
                if (textEditor.Target == propertyGrid.SelectedObject)
                    textEditor.Text = textEditor.Target.Name;
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

        private void RefreshNode(MyWorkingNode node)
        {
            node.Updated();

            propertyGrid.Refresh();
            m_mainForm.InvalidateGraphLayouts();
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

            MyWorkingNode node = Target as MyWorkingNode;
            if (node == null)
                return;

            node.SaveOnStop = saveNodeDataButton.Checked;
                
            RefreshNode(node);
        }

        private void loadNodeDataButton_Click(object sender, EventArgs e)
        {
            MyWorkingNode node = Target as MyWorkingNode;
            if (node == null)
                return;

            node.LoadOnStart = loadNodeDataButton.Checked;

            RefreshNode(node);
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

        private void dashboardButton_CheckedChanged(object sender, EventArgs e)
        {
            if (propertyGrid.SelectedGridItem.PropertyDescriptor == null)
                return;

            PropertyDescriptor propertyDescriptor = propertyGrid.SelectedGridItem.PropertyDescriptor;

            if (propertyDescriptor != null)
                m_mainForm.DashboardPropertyToggle(Target, propertyDescriptor.Name, dashboardButton.Checked);
        }

        public void RefreshView()
        {
            propertyGrid.Refresh();
        }

        private void propertyGrid_SelectedGridItemChanged(object sender, SelectedGridItemChangedEventArgs e)
        {
            RefreshDashboardButton();
        }

        private void propertyGrid_Enter(object sender, EventArgs e)
        {
            RefreshDashboardButton();
        }

        private void RefreshDashboardButton()
        {
            if (ActiveControl == propertyGrid && propertyGrid.SelectedGridItem != null && Target is MyNode)
            {
                PropertyDescriptor descriptor = propertyGrid.SelectedGridItem.PropertyDescriptor;
                if (descriptor == null)
                    return;

                if (descriptor.IsReadOnly)
                {
                    dashboardButton.Enabled = false;
                    return;
                }

                // A real property has been selected.
                dashboardButton.Enabled = true;
                dashboardButton.Checked = m_mainForm.CheckDashboardContains(Target, descriptor.Name);
            }
            else
            {
                dashboardButton.Enabled = false;
            }
        }
    }
}
