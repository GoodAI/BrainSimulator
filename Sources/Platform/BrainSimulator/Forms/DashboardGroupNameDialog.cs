using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using GoodAI.Core.Dashboard;

namespace GoodAI.BrainSimulator.Forms
{
    public partial class DashboardGroupNameDialog : Form
    {
        private readonly PropertyGrid m_sourceGrid;
        private readonly DashboardPropertyGroup m_group;
        private readonly GroupDashboard m_dashboard;

        public DashboardGroupNameDialog(PropertyGrid sourceGrid, DashboardPropertyGroup group, GroupDashboard dashboard)
        {
            m_sourceGrid = sourceGrid;
            m_group = @group;
            m_dashboard = dashboard;
            InitializeComponent();
            groupNameText.Text = group.PropertyName;
        }

        private void okButton_Click(object sender, EventArgs e)
        {
            SaveAndClose();
        }

        private void SaveAndClose()
        {
            string newName = groupNameText.Text;

            if (!m_dashboard.CanChangeName(m_group, newName))
                return;

            m_group.PropertyName = newName;
            m_sourceGrid.Refresh();
            Close();
        }

        private void groupNameText_KeyUp(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Enter)
                SaveAndClose();
        }

        private void groupNameText_TextChanged(object sender, EventArgs e)
        {
            okButton.Enabled = m_dashboard.CanChangeName(m_group, groupNameText.Text);
        }
    }
}
