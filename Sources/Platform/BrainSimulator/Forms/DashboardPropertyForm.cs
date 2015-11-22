using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Media;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using GoodAI.BrainSimulator.DashboardUtils;
using GoodAI.Core.Dashboard;
using GoodAI.Core.Nodes;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.BrainSimulator.Forms
{
    public partial class DashboardPropertyForm : DockContent
    {
        private MainForm m_mainForm;

        public event PropertyValueChangedEventHandler PropertyValueChanged
        {
            add { propertyGrid.PropertyValueChanged += value; }
            remove { propertyGrid.PropertyValueChanged -= value; }
        }

        public DashboardPropertyForm(MainForm mainForm)
        {
            m_mainForm = mainForm;
            InitializeComponent();
            DisableGroupButtons();
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

        public void UpdateDashboards(Dashboard dashboard, GroupDashboard groupedDashboard)
        {
            DashboardViewModel = new DashboardViewModel(dashboard);
            GroupedDashboardViewModel = new GroupedDashboardViewModel(groupedDashboard);
        }

        public bool CanEditNodeProperties
        {
            set
            {
                foreach (var propertyDescriptor in DashboardViewModel.GetProperties(new Attribute[0]))
                {
                    var proxyPropertyDescriptor = propertyDescriptor as ProxyPropertyDescriptor;
                    var property = proxyPropertyDescriptor.Property as SingleProxyProperty;
                    if (property != null && property.Target is MyNode)
                    {
                        proxyPropertyDescriptor.Property.ReadOnly = !value;
                    }
                    else
                    {
                        // TODO(HonzaS): Finish this.
                    }
                }

                propertyGrid.Refresh();
            }
        }

        private void OnDashboardPropertiesChanged(object sender, EventArgs args)
        {
            removeButton.Enabled = false;
            propertyGrid.Refresh();
        }

        private void OnGroupedDashboardPropertiesChanged(object sender, EventArgs args)
        {
            DisableGroupButtons();
            propertyGridGrouped.Refresh();
        }

        private void DisableGroupButtons()
        {
            removeGroupButton.Enabled = false;
            editGroupButton.Enabled = false;
            addToGroupButton.Enabled = false;
        }

        private void removeButton_Click(object sender, EventArgs e)
        {
            var descriptor = propertyGrid.SelectedGridItem.PropertyDescriptor as ProxyPropertyDescriptor;
            if (descriptor == null)
                throw new InvalidOperationException("Invalid property descriptor used in the dashboard.");

            DashboardViewModel.RemoveProperty(descriptor.Property);
        }

        private void propertyGrid_SelectedGridItemChanged(object sender, SelectedGridItemChangedEventArgs e)
        {
            if (GroupedDashboardViewModel == null)
                return;

            if (e.NewSelection != null)
            {
                removeButton.Enabled = true;
            }
        }

        private void addGroupButton_Click(object sender, EventArgs e)
        {
            GroupedDashboardViewModel.AddGroupedProperty();
        }

        private void removeGroupButton_Click(object sender, EventArgs e)
        {
            var descriptor = propertyGridGrouped.SelectedGridItem.PropertyDescriptor as ProxyPropertyGroupDescriptor;
            if (descriptor == null)
                throw new InvalidOperationException("Invalid property descriptor used in the dashboard.");

            GroupedDashboardViewModel.RemoveProperty(descriptor.Property);
            propertyGrid.Refresh();
        }

        private void editGroupButton_Click(object sender, EventArgs e)
        {
            var descriptor = propertyGridGrouped.SelectedGridItem.PropertyDescriptor as ProxyPropertyGroupDescriptor;
            if (descriptor == null)
                throw new InvalidOperationException("Invalid property descriptor used in the dashboard.");

            var dialog = new DashboardGroupNameDialog(propertyGridGrouped, descriptor.Property.SourceProperty);
            dialog.ShowDialog();
        }

        private void addToGroupButton_Click(object sender, EventArgs e)
        {
            // This can be null if the category was selected.
            var selectedPropertyDescriptor = propertyGrid.SelectedGridItem.PropertyDescriptor as ProxyPropertyDescriptor;
            if (selectedPropertyDescriptor == null)
                return;

            var property = selectedPropertyDescriptor.Property.SourceProperty;

            var selectedGroupDescriptor = propertyGridGrouped.SelectedGridItem.PropertyDescriptor as ProxyPropertyGroupDescriptor;
            var groupProperty = selectedGroupDescriptor.Property.SourceProperty;

            try
            {
                groupProperty.CheckType(property);
            }
            catch (InvalidOperationException exception)
            {
                // TODO(HonzaS): display an error.
            }

            groupProperty.Add(property);

            propertyGrid.Refresh();
            propertyGridGrouped.Refresh();
        }

        private void propertyGridGrouped_SelectedGridItemChanged(object sender, SelectedGridItemChangedEventArgs e)
        {
            if (e.NewSelection != null)
            {
                removeGroupButton.Enabled = true;
                editGroupButton.Enabled = true;
                addToGroupButton.Enabled = true;
            }
            propertyGrid.Refresh();
        }
    }
}
