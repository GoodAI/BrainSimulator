using GoodAI.BrainSimulator.NodeView;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System.ComponentModel;
using System.Windows.Forms;
using GoodAI.Core.Dashboard;
using GoodAI.Core.Nodes;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.BrainSimulator.Forms
{
    public partial class TaskPropertyForm : DockContent, INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;

        protected virtual void OnPropertyChanged(string propertyName = null)
        {
            if (PropertyChanged != null)
                PropertyChanged(this, new PropertyChangedEventArgs(propertyName));

            m_mainForm.ProjectStateChanged(string.Format("Task property value changed: {0}", propertyName));
        }

        private MainForm m_mainForm;

        public MyTask Target
        {
            get { return propertyGrid.SelectedObject as MyTask; }
            set { propertyGrid.SelectedObject = value; }
        }

        public TaskPropertyForm(MainForm mainForm)
        {
            m_mainForm = mainForm;
            InitializeComponent();                   
            propertyGrid.BrowsableAttributes = new AttributeCollection(new MyBrowsableAttribute());
        }

        private void propertyGrid_PropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            OnPropertyChanged(e.ChangedItem.PropertyDescriptor.Name);

            Target.GenericOwner.Updated();
        }

        public void RefreshView()
        {
            propertyGrid.Refresh();
        }

        private void dashboardButton_CheckedChanged(object sender, System.EventArgs e)
        {
            PropertyDescriptor propertyDescriptor = propertyGrid.SelectedGridItem.PropertyDescriptor;

            if (propertyDescriptor != null)
                m_mainForm.DashboardPropertyToggle(Target, propertyDescriptor.Name, dashboardButton.Checked);
        }

        private void propertyGrid_SelectedGridItemChanged(object sender, SelectedGridItemChangedEventArgs e)
        {
            RefreshDashboardButton();
        }

        private void propertyGrid_Enter(object sender, System.EventArgs e)
        {
            RefreshDashboardButton();
        }

        private void RefreshDashboardButton()
        {
            if (ActiveControl == propertyGrid && propertyGrid.SelectedGridItem != null && Target is MyTask)
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
