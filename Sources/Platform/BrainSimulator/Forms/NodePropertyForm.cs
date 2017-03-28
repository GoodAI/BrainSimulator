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
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));

            m_mainForm.ProjectStateChanged($"Node property value changed: {propertyName}");
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
            set
            {
                propertyGrid.SelectedObject = value;
                UpdateUi();
            }
        }

        public object[] Targets
        {
            get { return propertyGrid.SelectedObjects; }
            set
            {
                propertyGrid.SelectedObjects = value;
                UpdateUi();
            }
        }

        private bool HasMultipleTargets => (Targets?.Length ?? 0) > 1;

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


        private void UpdateUi()
        {
            UpdateTitleAndButtons();
            UpdateObserverList();
        }

        private void UpdateTitleAndButtons()
        {
            if (HasMultipleTargets)
            {
                nodeNameTextBox.Rtf = @"{\rtf1\ansi \b " + Targets?.Length + @" \b0 nodes selected.}";
            }
            else if (Target is MyNode)
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

            // TODO(Premek): Allow help for multiple nodes of the same type.
            helpButton.Enabled = !HasMultipleTargets && (Target is MyWorkingNode || Target is MyAbstractObserver);

            snapshotButton.Enabled = !HasMultipleTargets && (Target is MyAbstractObserver);

            // TODO(Premek): Allow to set SaveOnStop / LoadOnStart for multiple nodes.
            var workingNode = !HasMultipleTargets ? (Target as MyWorkingNode) : null;
            var isWorkingNode = (workingNode != null);

            saveNodeDataButton.Enabled = isWorkingNode;
            saveNodeDataButton.Checked = workingNode?.SaveOnStop ?? false;

            loadNodeDataButton.Enabled = isWorkingNode;
            loadNodeDataButton.Checked = workingNode?.LoadOnStart ?? false;

            clearDataButton.Enabled = isWorkingNode;

            dashboardButton.Enabled = !HasMultipleTargets && (Target is MyNode);
        }

        private void UpdateObserverList()
        {
            observerDropDownButton.DropDownItems.Clear();
            observerDropDownButton.Enabled = false;

            if (HasMultipleTargets)
                return;

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
                    ToolStripMenuItem item = new ToolStripMenuItem(GetMenuItemName(oc));
                    item.Tag = oc.ObserverType;
                    item.Click += item_Click;
                    observerDropDownButton.DropDownItems.Add(item);
                }
                observerDropDownButton.Enabled = true;
            }            
        }

        private string GetMenuItemName(MyObserverConfig oc)
        {
            if (oc.ObserverType.Name.Substring(0, 2).Equals("My"))
            {
                return oc.ObserverType.Name.Substring(2);
            }
            return oc.ObserverType.Name;
        }

        private void item_Click(object sender, EventArgs e)
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
            var propertyDescriptor = propertyGrid.SelectedGridItem.PropertyDescriptor;
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
