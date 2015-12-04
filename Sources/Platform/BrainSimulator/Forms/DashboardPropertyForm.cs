using System;
using System.ComponentModel;
using System.Linq;
using System.Windows.Forms;
using GoodAI.BrainSimulator.DashboardUtils;
using GoodAI.Core.Dashboard;
using GoodAI.Core.Execution;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.BrainSimulator.Forms
{
    public partial class DashboardPropertyForm : DockContent
    {
        private readonly MainForm m_mainForm;

        public event PropertyValueChangedEventHandler PropertyValueChanged
        {
            add
            {
                propertyGrid.PropertyValueChanged += value;
                propertyGridGrouped.PropertyValueChanged += value;
            }
            remove
            {
                propertyGrid.PropertyValueChanged -= value;
                propertyGridGrouped.PropertyValueChanged -= value;
            }
        }

        public DashboardPropertyForm(MainForm mainForm)
        {
            m_mainForm = mainForm;
            InitializeComponent();
            DisableGroupButtons();

            propertyGrid.PropertyValueChanged += OnPropertyValueChanged;
            propertyGridGrouped.PropertyValueChanged += OnGroupPropertyValueChanged;
        }

        private void OnGroupPropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            ProxyPropertyGroupDescriptor descriptor = GetCurrentGroupDescriptor();
            foreach (MyNode node in descriptor.Proxy.SourceProperty.GroupedProperties.Select(member => member.Node))
                RefreshNode(node);

            RefreshAll();
        }

        private void OnPropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            ProxyPropertyDescriptor descriptor = GetCurrentPropertyDescriptor();
            MyNode node = descriptor.Proxy.SourceProperty.Node;
            RefreshNode(node);

            RefreshAll();
        }

        private void RefreshAll()
        {
            propertyGrid.Refresh();
            propertyGridGrouped.Refresh();
            memberListBox.Refresh();
        }

        private void RefreshNode(MyNode node)
        {
            node.Updated();

            propertyGrid.Refresh();
            m_mainForm.InvalidateGraphLayouts();
        }

        private DashboardViewModel DashboardViewModel
        {
            get { return propertyGrid.SelectedObject as DashboardViewModel; }
            set
            {
                if (DashboardViewModel != null)
                    DashboardViewModel.PropertyChanged -= OnDashboardPropertiesChanged;

                propertyGrid.SelectedObject = value;
                value.PropertyChanged += OnDashboardPropertiesChanged;
            }
        }

        private GroupedDashboardViewModel GroupedDashboardViewModel
        {
            get { return propertyGridGrouped.SelectedObject as GroupedDashboardViewModel; }
            set
            {
                if (GroupedDashboardViewModel != null)
                    GroupedDashboardViewModel.PropertyChanged -= OnGroupedDashboardPropertiesChanged;

                propertyGridGrouped.SelectedObject = value;
                value.PropertyChanged += OnGroupedDashboardPropertiesChanged;
            }
        }

        public void SetDashboards(Dashboard dashboard, GroupDashboard groupedDashboard)
        {
            DashboardViewModel = new DashboardViewModel(dashboard);
            GroupedDashboardViewModel = new GroupedDashboardViewModel(groupedDashboard);
        }

        private bool CanEditNodeProperties
        {
            set
            {
                foreach (SingleProxyProperty propertyProxy in DashboardViewModel.GetProperties(new Attribute[0])
                    .Cast<ProxyPropertyDescriptor>()
                    .Select(descriptor => descriptor.Proxy)
                    .Where(proxy => proxy.Target is MyNode))
                {
                    propertyProxy.ReadOnly = !value;
                }

                foreach (ProxyPropertyGroup groupProxy in GroupedDashboardViewModel.GetProperties(new Attribute[0])
                    .Cast<ProxyPropertyGroupDescriptor>()
                    .Select(descriptor => descriptor.Proxy)
                    .Where(proxy => proxy.SourceProperty.GroupedProperties.Any(property => property.Target is MyNode)))
                {
                    groupProxy.ReadOnly = !value;
                }

                RefreshAll();
            }
        }

        private void OnDashboardPropertiesChanged(object sender, EventArgs args)
        {
            SetPropertyGridButtonsEnabled(false);
            RefreshAll();
        }

        private void SetPropertyGridButtonsEnabled(bool enabled)
        {
            removeButton.Enabled = enabled;
            goToNodeButton.Enabled = enabled;
        }

        private void OnGroupedDashboardPropertiesChanged(object sender, EventArgs args)
        {
            DisableGroupButtons();
            RefreshAll();
        }

        private void DisableGroupButtons()
        {
            SetPropertyGridGroupedButtonsEnabled(false);
        }

        private void SetPropertyGridGroupedButtonsEnabled(bool enabled)
        {
            removeGroupButton.Enabled = enabled;
            editGroupButton.Enabled = enabled;
            addToGroupButton.Enabled = enabled;
            removeFromGroupButton.Enabled = enabled;
        }

        private void removeButton_Click(object sender, EventArgs e)
        {
            ProxyPropertyDescriptor descriptor = GetCurrentPropertyDescriptor();

            DashboardViewModel.RemoveProperty(descriptor.Proxy);

            // TODO: Undo
        }

        private void propertyGrid_SelectedGridItemChanged(object sender, SelectedGridItemChangedEventArgs e)
        {
            ClearError();

            if (GroupedDashboardViewModel == null)
                return;

            if (e.NewSelection != null)
                SetPropertyGridButtonsEnabled(true);
        }

        private void ClearError()
        {
            errorText.Text = "";
        }

        private void addGroupButton_Click(object sender, EventArgs e)
        {
            GroupedDashboardViewModel.AddGroupedProperty();

            // TODO: Undo
        }

        private void removeGroupButton_Click(object sender, EventArgs e)
        {
            ProxyPropertyGroupDescriptor descriptor = GetCurrentGroupDescriptor();

            GroupedDashboardViewModel.RemoveProperty(descriptor.Proxy);
            propertyGrid.Refresh();
            memberListBox.Items.Clear();

            // TODO: Undo
        }

        private ProxyPropertyDescriptor GetCurrentPropertyDescriptor()
        {
            var descriptor = propertyGrid.SelectedGridItem.PropertyDescriptor as ProxyPropertyDescriptor;
            if (descriptor == null)
                throw new InvalidOperationException("Invalid property descriptor used in the dashboard.");
            return descriptor;
        }

        private ProxyPropertyGroupDescriptor GetCurrentGroupDescriptor()
        {
            var descriptor =
                propertyGridGrouped.SelectedGridItem.PropertyDescriptor as ProxyPropertyGroupDescriptor;
            if (descriptor == null)
                throw new InvalidOperationException("The group property grid contained an invalid descriptor.");

            return descriptor;
        }

        private void editGroupButton_Click(object sender, EventArgs e)
        {
            ProxyPropertyGroupDescriptor descriptor = GetCurrentGroupDescriptor();

            var dialog = new DashboardGroupNameDialog(propertyGridGrouped, descriptor.Proxy.SourceProperty);
            dialog.ShowDialog();
        }

        private void addToGroupButton_Click(object sender, EventArgs e)
        {
            // This can be null if the category was selected.
            var selectedPropertyDescriptor = propertyGrid.SelectedGridItem.PropertyDescriptor as ProxyPropertyDescriptor;
            if (selectedPropertyDescriptor == null)
                return;

            DashboardNodeProperty property = selectedPropertyDescriptor.Proxy.SourceProperty;

            ProxyPropertyGroupDescriptor selectedGroupDescriptor = GetCurrentGroupDescriptor();
            DashboardPropertyGroup groupProperty = selectedGroupDescriptor.Proxy.SourceProperty;

            try
            {
                groupProperty.Add(property);

                SetPropertyGridButtonsEnabled(false);
                RefreshAll();
            }
            catch (InvalidOperationException)
            {
                errorText.Text = string.Format("Cannot add a {0} property to a {1} group",
                    selectedPropertyDescriptor.PropertyType.Name,
                    selectedGroupDescriptor.PropertyType.Name);
            }
        }

        private void propertyGridGrouped_SelectedGridItemChanged(object sender, SelectedGridItemChangedEventArgs e)
        {
            ClearError();

            memberListBox.Items.Clear();
            if (e.NewSelection != null)
            {
                SetPropertyGridGroupedButtonsEnabled(true);

                LoadGroupedProperties(e.NewSelection.PropertyDescriptor as ProxyPropertyGroupDescriptor);
            }
            propertyGrid.Refresh();
        }

        private void LoadGroupedProperties(ProxyPropertyGroupDescriptor groupDescriptor)
        {
            foreach (var proxy in groupDescriptor.Proxy.GetGroupMembers())
                memberListBox.Items.Add(proxy);
        }

        private void removeFromGroupButton_Click(object sender, EventArgs e)
        {
            foreach (var proxy in memberListBox.SelectedItems.Cast<SingleProxyProperty>())
                proxy.SourceProperty.Group.Remove(proxy.SourceProperty);

            RefreshAll();
        }

        public void OnSimulationStateChanged(object sender, MySimulationHandler.StateEventArgs e)
        {
            CanEditNodeProperties = e.NewState == MySimulationHandler.SimulationState.STOPPED;
        }

        private void goToNodeButton_Click(object sender, EventArgs e)
        {
            SingleProxyProperty proxy = GetCurrentPropertyDescriptor().Proxy;
            GoToNode(sender, proxy);
        }

        private void GoToNode(object sender, SingleProxyProperty proxy)
        {
            MyNode targetNode = proxy.SourceProperty.Node;

            if (targetNode is MyWorld)
            {
                GraphLayoutForm graphForm = m_mainForm.OpenGraphLayout(targetNode.Owner.Network);
                graphForm.worldButton_Click(sender, EventArgs.Empty);
            }
            else
            {
                GraphLayoutForm graphForm = m_mainForm.OpenGraphLayout(targetNode.Parent);
                graphForm.SelectNodeView(targetNode);
            }
        }

        private void goToNodeFromMemberButton_Click(object sender, EventArgs e)
        {
            SingleProxyProperty proxy = memberListBox.SelectedItems.Cast<SingleProxyProperty>().FirstOrDefault();
            if (proxy != null)
                GoToNode(sender, proxy);
        }

        public void OnPropertyExternallyChanged(object sender, PropertyChangedEventArgs e)
        {
            // If the property is grouped, replace its value by whatever is set in the group.

            propertyGrid.Refresh();
            object target;
            var nodeSender = sender as NodePropertyForm;
            if (nodeSender != null)
            {
                target = nodeSender.Target;
            }
            else
            {
                var taskSender = sender as TaskPropertyForm;
                target = taskSender.Target;
            }

            PreserveGroupValue(e.PropertyName, target);

            propertyGrid.Refresh();
        }

        private void PreserveGroupValue(string propertyName, object target)
        {
            DashboardNodeProperty property = DashboardViewModel.GetProperty(target, propertyName);
            if (property == null)
                return;

            DashboardPropertyGroup group = property.Group;
            if (group == null)
                return;

            object valueOfGroupMembers = @group.GroupedProperties
                .Select(member => member.Proxy.Value)
                .FirstOrDefault(value => !value.Equals(property.Proxy.Value));

            if (valueOfGroupMembers == null)
                return;
            
            MyLog.WARNING.WriteLine("Trying to change a group property {0}. Value reverted to {1}.", propertyName,
                valueOfGroupMembers);
            property.Proxy.Value = valueOfGroupMembers;
        }

        private void showGroupsButton_CheckedChanged(object sender, EventArgs e)
        {
            ShowHideGroups();
        }

        private void showMembersButton_CheckedChanged(object sender, EventArgs e)
        {
            ShowHideGroups();
        }

        private void ShowHideGroups()
        {
            splitContainerProperties.Panel2Collapsed = !(showGroupsButton.Checked || showMembersButton.Checked);

            if (splitContainerProperties.Panel2Collapsed)
                return;

            splitContainerGroups.Panel1Collapsed = !showGroupsButton.Checked;
            splitContainerGroups.Panel2Collapsed = !showMembersButton.Checked;
        }

        public void RemovePropertiesOfNode(MyNode node)
        {
            // First try to remove the property from a group.
            RemovePropertiesOfNode(node, GroupedDashboardViewModel);
            RemovePropertiesOfNode(node, DashboardViewModel);

            RefreshAll();
        }

        private void RemovePropertiesOfNode<TDashboard, TProperty>(MyNode node, DashboardViewModelBase<TDashboard, TProperty> model)
            where TDashboard : DashboardBase<TProperty>
            where TProperty : DashboardProperty
        {
            model.RemovePropertyOf(node);
        }
    }
}
