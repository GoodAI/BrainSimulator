using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
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
        }

        public DashboardViewModel Target
        {
            private get { return propertyGrid.SelectedObject as DashboardViewModel; }
            set
            {
                if (Target != null)
                    Target.PropertyChanged -= OnTargetPropertiesChanged;

                propertyGrid.SelectedObject = value;
                value.PropertyChanged += OnTargetPropertiesChanged;
            }
        }

        public bool CanEditNodeProperties
        {
            set
            {
                foreach (var propertyDescriptor in Target.GetProperties(new Attribute[0]))
                {
                    var proxyPropertyDescriptor = propertyDescriptor as ProxyPropertyDescriptor;
                    if (proxyPropertyDescriptor.Property.Owner is MyNode)
                    {
                        proxyPropertyDescriptor.Property.ReadOnly = !value;
                    }
                }

                propertyGrid.Refresh();
            }
        }

        private void OnTargetPropertiesChanged(object sender, EventArgs args)
        {
            removeButton.Enabled = false;
            propertyGrid.Refresh();
        }

        private void removeButton_Click(object sender, EventArgs e)
        {
            var descriptor = propertyGrid.SelectedGridItem.PropertyDescriptor as ProxyPropertyDescriptor;
            if (descriptor == null)
                throw new InvalidOperationException("Invalid property descriptor used in the dashboard.");

            Target.RemoveProperty(descriptor.Property.Owner, descriptor.Property.PropertyInfo.Name);
        }

        private void propertyGrid_SelectedGridItemChanged(object sender, SelectedGridItemChangedEventArgs e)
        {
            if (e.NewSelection != null)
            {
                removeButton.Enabled = true;
            }
        }
    }
}
