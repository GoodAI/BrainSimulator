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
        private DashboardPropertyGroup m_group;

        public DashboardGroupNameDialog(PropertyGrid sourceGrid, DashboardPropertyGroup group)
        {
            m_sourceGrid = sourceGrid;
            m_group = @group;
            InitializeComponent();
            groupNameText.Text = group.PropertyName;
        }

        private void okButton_Click(object sender, EventArgs e)
        {
            SaveAndClose();
        }

        private void SaveAndClose()
        {
            m_group.PropertyName = groupNameText.Text;
            m_sourceGrid.Refresh();
            Close();
        }

        private void groupNameText_KeyUp(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Enter)
                SaveAndClose();
        }
    }
}
