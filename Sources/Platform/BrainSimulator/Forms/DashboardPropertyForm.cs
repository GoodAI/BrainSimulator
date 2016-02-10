using System;
using System.ComponentModel;
using System.Linq;
using System.Windows.Forms;
using GoodAI.BrainSimulator.DashboardUtils;
using GoodAI.Core.Dashboard;
using GoodAI.Core.Execution;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
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
            ProxyPropertyDescriptor descriptor = GetCurrentGroupDescriptor();
            foreach (MyNode node in GroupedDashboardViewModel.GetProperty(descriptor.Proxy.PropertyId).GroupedProperties.Select(member => member.Node))
                RefreshNode(node);

            RefreshAll();
        }

        private void OnPropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            ProxyPropertyDescriptor descriptor = GetCurrentPropertyDescriptor();
            MyNode node = DashboardViewModel.GetProperty(descriptor.Proxy.PropertyId).Node;
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
                    .OfType<SingleProxyProperty>()
                    .Where(proxy => proxy.Target is MyNode))
                {
                    propertyProxy.ReadOnly = !value;
                }

                foreach (ProxyPropertyGroup groupProxy in GroupedDashboardViewModel.GetProperties(new Attribute[0])
                    .Cast<ProxyPropertyDescriptor>()
                    .Select(descriptor => descriptor.Proxy)
                    .OfType<ProxyPropertyGroup>()
                    .Where(proxy => GroupedDashboardViewModel.GetProperty(proxy.PropertyId)
                                    .GroupedProperties.Any(property => !(property is DashboardTaskProperty))))
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

            m_mainForm.ProjectStateChanged(string.Format("Dashboard property removed: {0}", descriptor.Name));
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

            m_mainForm.ProjectStateChanged("Dashboard property group added");
        }

        private void removeGroupButton_Click(object sender, EventArgs e)
        {
            ProxyPropertyDescriptor descriptor = GetCurrentGroupDescriptor();

            GroupedDashboardViewModel.RemoveProperty(descriptor.Proxy);
            propertyGrid.Refresh();
            memberListBox.Items.Clear();

            m_mainForm.ProjectStateChanged(string.Format("Dashboard property group removed: {0}", descriptor.Name));
        }

        private ProxyPropertyDescriptor GetCurrentPropertyDescriptor()
        {
            return GetCurrentDescriptorForGrid(propertyGrid, errorMessage: "Invalid property descriptor used in the dashboard.");
        }

        private ProxyPropertyDescriptor GetCurrentGroupDescriptor()
        {
            return GetCurrentDescriptorForGrid(propertyGridGrouped, errorMessage: "The group property grid contained an invalid descriptor.");
        }

        private ProxyPropertyDescriptor GetCurrentDescriptorForGrid(PropertyGrid grid, string errorMessage = "")
        {
            var descriptor = grid.SelectedGridItem.PropertyDescriptor as ProxyPropertyDescriptor;
            if (descriptor == null)
                throw new InvalidOperationException(errorMessage);

            return descriptor;
        }

        private void editGroupButton_Click(object sender, EventArgs e)
        {
            ProxyPropertyDescriptor descriptor = GetCurrentGroupDescriptor();

            var dialog = new DashboardGroupNameDialog(propertyGridGrouped,
                GroupedDashboardViewModel.GetProperty(descriptor.Proxy.PropertyId),
                m_mainForm.Project.GroupedDashboard);

            dialog.ShowDialog();
        }

        private void addToGroupButton_Click(object sender, EventArgs e)
        {
            // This can be null if the category was selected.
            var selectedPropertyDescriptor = propertyGrid.SelectedGridItem.PropertyDescriptor as ProxyPropertyDescriptor;
            if (selectedPropertyDescriptor == null)
                return;

            DashboardNodePropertyBase property = DashboardViewModel.GetProperty(selectedPropertyDescriptor.Proxy.PropertyId);

            ProxyPropertyDescriptor selectedGroupDescriptor = GetCurrentGroupDescriptor();
            DashboardPropertyGroup groupProperty =
                GroupedDashboardViewModel.GetProperty(selectedGroupDescriptor.Proxy.PropertyId);

            try
            {
                groupProperty.Add(property);

                SetPropertyGridButtonsEnabled(false);
                RefreshAll();

                m_mainForm.ProjectStateChanged(string.Format("Dashboard property {0} added to group: {1}",
                    property.DisplayName, groupProperty.DisplayName));
            }
            catch (InvalidOperationException)
            {
                errorText.Text = string.Format("Cannot add a {0} property to a {1} group",
                    selectedPropertyDescriptor.Proxy.TypeName,
                    selectedGroupDescriptor.Proxy.TypeName);
            }
        }

        private void propertyGridGrouped_SelectedGridItemChanged(object sender, SelectedGridItemChangedEventArgs e)
        {
            ClearError();

            memberListBox.Items.Clear();
            if (e.NewSelection != null)
            {
                SetPropertyGridGroupedButtonsEnabled(true);

                LoadGroupedProperties(e.NewSelection.PropertyDescriptor as ProxyPropertyDescriptor);
            }
            propertyGrid.Refresh();
        }

        private void LoadGroupedProperties(ProxyPropertyDescriptor groupDescriptor)
        {
            foreach (ProxyPropertyBase proxy in GroupedDashboardViewModel.GetProperty(groupDescriptor.Proxy.PropertyId)
                    .GroupedProperties.Select(property => property.GenericProxy))
                memberListBox.Items.Add(proxy);
        }

        private void removeFromGroupButton_Click(object sender, EventArgs e)
        {
            foreach (DashboardNodePropertyBase property in
                    memberListBox.SelectedItems.Cast<ProxyPropertyBase>()
                        .Select(proxy => DashboardViewModel.GetProperty(proxy.PropertyId)))
            {
                property.Group.Remove(property);
            }

            m_mainForm.ProjectStateChanged(string.Format("Dashboard property removed from group: {0}",
                GetCurrentGroupDescriptor().Name));

            RefreshAll();
        }

        public void OnSimulationStateChanged(object sender, MySimulationHandler.StateEventArgs e)
        {
            CanEditNodeProperties = e.NewState == MySimulationHandler.SimulationState.STOPPED;
        }

        private void goToNodeButton_Click(object sender, EventArgs e)
        {
            var property = DashboardViewModel.GetProperty(GetCurrentPropertyDescriptor().Proxy.PropertyId);
            GoToNode(sender, property.Node);
        }

        private void GoToNode(object sender, MyNode node)
        {
            if (node is MyWorld)
            {
                GraphLayoutForm graphForm = m_mainForm.OpenGraphLayout(node.Owner.Network);
                graphForm.worldButton_Click(sender, EventArgs.Empty);
            }
            else
            {
                GraphLayoutForm graphForm = m_mainForm.OpenGraphLayout(node.Parent);
                graphForm.SelectNodeView(node);
            }
        }

        private void goToNodeFromMemberButton_Click(object sender, EventArgs e)
        {
            ProxyPropertyBase proxy = memberListBox.SelectedItems.Cast<ProxyPropertyBase>().FirstOrDefault();
            if (proxy == null)
                return;

            MyNode node = DashboardViewModel.GetProperty(proxy.PropertyId).Node;
            GoToNode(sender, node);
        }

        public void OnPropertyExternallyChanged(object sender, PropertyChangedEventArgs e)
        {
            // If the property is grouped, replace its value by whatever is set in the group.

            propertyGrid.Refresh();
            object target = null;
            var nodeSender = sender as NodePropertyForm;
            if (nodeSender != null)
                target = nodeSender.Target;

            var taskPropertySender = sender as TaskPropertyForm;
            if (taskPropertySender != null)
                target = taskPropertySender.Target;

            var taskSender = sender as MyTask;
            if (taskSender != null)
                target = taskSender;

            var taskGroupSender = sender as TaskGroup;
            if (taskGroupSender != null)
                target = taskGroupSender;
            
            if (target != null)
            {
                PreserveGroupValue(e.PropertyName, target);
                propertyGrid.Refresh();
                propertyGridGrouped.Refresh();
            }
        }

        private void PreserveGroupValue(string propertyName, object target)
        {
            DashboardNodePropertyBase property = DashboardViewModel.GetProperty(target, propertyName);
            if (property == null)
                return;

            DashboardPropertyGroup group = property.Group;
            if (group == null)
                return;

            object valueOfGroupMembers = @group.GroupedProperties
                .Select(member => member.GenericProxy.Value)
                .FirstOrDefault(value => !value.Equals(property.GenericProxy.Value));

            if (valueOfGroupMembers == null)
                return;
            
            MyLog.WARNING.WriteLine("Trying to change a group property {0}. Value reverted to {1}.", propertyName,
                valueOfGroupMembers);
            property.GenericProxy.Value = valueOfGroupMembers;
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
