using GoodAI.BrainSimulator.NodeView;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System.ComponentModel;
using System.Linq;
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
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));

            m_mainForm.ProjectStateChanged($"Task property value changed: {propertyName}");
        }

        private readonly MainForm m_mainForm;

        public MyTask Target
        {
            get { return propertyGrid.SelectedObject as MyTask; }
            set { propertyGrid.SelectedObject = value; }
        }

        public object[] Targets
        {
            get { return propertyGrid.SelectedObjects; }
            set { propertyGrid.SelectedObjects = value; }
        }

        private bool HasMultipleTargets => (Targets?.Length ?? 0) > 1;

        public TaskPropertyForm(MainForm mainForm)
        {
            m_mainForm = mainForm;
            InitializeComponent();                   
            propertyGrid.BrowsableAttributes = new AttributeCollection(new MyBrowsableAttribute());
        }

        private void propertyGrid_PropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            OnPropertyChanged(e.ChangedItem.PropertyDescriptor?.Name ?? "(null)");

            foreach (var task in Targets.Select(it => it as MyTask))
            {
                task?.GenericOwner.Updated();
            }
        }

        public void RefreshView()
        {
            propertyGrid.Refresh();
        }

        private void dashboardButton_CheckedChanged(object sender, System.EventArgs e)
        {
            if (HasMultipleTargets)
            {
                MyLog.WARNING.WriteLine($"{nameof(dashboardButton)}: bulk toggle not implemented.");
                return;
            }

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
            if (ActiveControl != propertyGrid || propertyGrid.SelectedGridItem == null || Target == null || HasMultipleTargets)
            {
                dashboardButton.Enabled = false;
                return;
            }

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
    }
}
