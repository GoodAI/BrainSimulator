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

        private void OnTargetPropertiesChanged(object sender, EventArgs args)
        {
            propertyGrid.Refresh();
        }
    }
}
