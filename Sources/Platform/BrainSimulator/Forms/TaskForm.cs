using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;
using System.Windows.Forms.VisualStyles;
using GoodAI.BrainSimulator.Nodes;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.BrainSimulator.Forms
{
    public partial class TaskForm : DockContent, INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;

        private void OnPropertyChanged(object target, string propertyName = null)
        {
            PropertyChanged?.Invoke(target, new PropertyChangedEventArgs(propertyName));
        }

        private readonly MainForm m_mainForm;
        private bool m_isUpdating;

        private int m_lastSelectedTaskIndex = -1;

        private ListViewHitTestInfo m_lastHitTest;
        private const string EnabledPropertyName = "Enabled";

        public MyWorkingNode Target
        {
            set { Targets = new[] { value }; }
        }

        public IEnumerable<MyWorkingNode> Targets
        {
            get { return m_nodeSelection.Nodes; }
            set
            {
                StoreSelectedIndex();
                m_nodeSelection = new NodeSelection(value);
                RefreshView();
            }
        }

        private NodeSelection m_nodeSelection = NodeSelection.Empty;

        public TaskForm(MainForm mainForm)
        {
            m_mainForm = mainForm;

            InitializeComponent();
        }

        public void RefreshView()
        {
            m_isUpdating = true;

            listView.Items.Clear();
            if (!m_nodeSelection.IsEmpty)
            {
                foreach (var taskSelection in m_nodeSelection.Tasks)
                {
                    var item = new ListViewItem(new[] {taskSelection.Name, taskSelection.OneShot ? "Init" : ""})
                    {
                        Checked = taskSelection.AllEnabled,
                        Tag = taskSelection
                    };

                    listView.Items.Add(item);
                }
            }

            m_isUpdating = false;

            if (listView.Items.Count > m_lastSelectedTaskIndex)
            {
                listView.Items[m_lastSelectedTaskIndex].Selected = true;
                listView.Invalidate();
            }
            else
            {
                m_mainForm.TaskPropertyView.Target = null;
            }

            RefreshDashboardButton();
        }

        private static TaskSelection CastTag(object tag)
        {
            var result = tag as TaskSelection;

            if (result == null)
                MyLog.WARNING.WriteLine($"{nameof(TaskForm)}: Wrong tag!");

            return result;
        }

        private void StoreSelectedIndex()
        {
            m_lastSelectedTaskIndex = (listView.SelectedIndices.Count == 1) ? listView.SelectedIndices[0] : 0;
        }

        private void UpdateTasksEnableState()
        {
            m_isUpdating = true;

            foreach (ListViewItem item in listView.Items)
            {
                if (item == null)
                    continue;

                item.Checked = CastTag(item.Tag).AllEnabled;
            }

            m_isUpdating = false;
        }

        private void listView_ItemChecked(object sender, ItemCheckedEventArgs e)
        {
            if (m_isUpdating)
                return;

            var taskSelection = CastTag(e.Item.Tag);
            if (taskSelection == null)
                return;

            foreach (var task in taskSelection.EnumerateTasks().Where(t => !t.DesignTime))
            {
                task.Enabled = e.Item.Checked;

                object target = task;
                string propertyName = EnabledPropertyName;
                if (!string.IsNullOrEmpty(task.TaskGroupName))
                {
                    target = task.TaskGroup;
                    propertyName = task.TaskGroupName;
                }

                OnPropertyChanged(target, propertyName);

                task.GenericOwner.Updated();
            }

            UpdateTasksEnableState();
        }

        private void listView_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (listView.SelectedItems.Count == 1)
            {
                m_mainForm.TaskPropertyView.Targets = CastTag(listView.SelectedItems[0].Tag)?.ToObjectArray();
            }

            RefreshDashboardButton();
        }

        private static readonly Brush HL_BRUSH = new SolidBrush(SystemColors.Highlight);
        private static readonly Pen LINE_PEN = new Pen(Brushes.LightGray);

        private void listView_DrawSubItem(object sender, DrawListViewSubItemEventArgs e)
        {
            Rectangle bounds = e.SubItem.Bounds;

            if (e.ColumnIndex == 0)
                bounds.Width = bounds.X + e.Item.SubItems[1].Bounds.X;

            TaskSelection tasks = CastTag(e.Item.Tag);

            // Toggle colors if the item is highlighted.
            DrawBackgroundAndText(e, tasks.Forbidden);

            int xOffset = 0;

            if (e.ColumnIndex == 0)
            {
                DrawCheckBox(e, tasks, out xOffset);
            }
            else if (e.ColumnIndex == 1 && tasks.DesignTime)
            {                
                DrawPushButton(e, tasks.AllEnabled);
            }
            
            // Add a 2 pixel buffer to match the default behavior.
            var rec = new Rectangle(e.Bounds.X + 2 + xOffset, e.Bounds.Y + 2, e.Bounds.Width - 4, e.Bounds.Height - 4);

            // TODO: Confirm combination of TextFormatFlags.EndEllipsis and TextFormatFlags.ExpandTabs works on all systems.
            // MSDN claims they're exclusive but on Win7-64 they work.
            TextFormatFlags flags = TextFormatFlags.Left | TextFormatFlags.EndEllipsis | TextFormatFlags.ExpandTabs |
                                    TextFormatFlags.SingleLine;

            // If a different tabstop than the default is needed, will have to p/invoke DrawTextEx from win32.
            TextRenderer.DrawText(e.Graphics, e.SubItem.Text, e.Item.ListView.Font, rec, e.SubItem.ForeColor, flags);         
        }

        private static void DrawCheckBox(DrawListViewSubItemEventArgs e, TaskSelection tasks, out int checkboxWidth)
        {
            Point glyphPoint = new Point(4, e.Item.Position.Y + 2);

            if (!string.IsNullOrEmpty(tasks.TaskGroupName))
            {
                RadioButtonState state;
                if (tasks.Forbidden)
                    state = RadioButtonState.UncheckedDisabled;
                else
                    state = e.Item.Checked ? RadioButtonState.CheckedNormal : RadioButtonState.UncheckedNormal;

                RadioButtonRenderer.DrawRadioButton(e.Graphics, glyphPoint, state);
                checkboxWidth = RadioButtonRenderer.GetGlyphSize(e.Graphics, state).Width + 4;
            }
            else if (tasks.DesignTime)
            {
                checkboxWidth = CheckBoxRenderer.GetGlyphSize(e.Graphics, CheckBoxState.UncheckedNormal).Width + 4;
            }
            else
            {
                CheckBoxState state;
                if (tasks.Forbidden)
                    state = CheckBoxState.UncheckedDisabled;
                else
                    state = (tasks.Enabled3State == Enabled3State.AllEnabled)
                        ? CheckBoxState.CheckedNormal
                        : (tasks.Enabled3State == Enabled3State.AllDisabled)
                            ? CheckBoxState.UncheckedNormal
                            : CheckBoxState.MixedNormal;

                CheckBoxRenderer.DrawCheckBox(e.Graphics, glyphPoint, state);
                checkboxWidth = CheckBoxRenderer.GetGlyphSize(e.Graphics, state).Width + 4;
            }
        }

        private void DrawPushButton(DrawListViewSubItemEventArgs e, bool enabled)
        {
            PushButtonState buttonState = PushButtonState.Disabled;

            if (m_mainForm.SimulationHandler.CanStart && enabled)
            {
                buttonState =
                    m_lastHitTest != null && e.Item == m_lastHitTest.Item && e.SubItem == m_lastHitTest.SubItem
                        ? PushButtonState.Pressed
                        : PushButtonState.Normal;
            }

            ButtonRenderer.DrawButton(e.Graphics, e.Bounds, "Execute", listView.Font, false, buttonState);
        }

        private static void DrawBackgroundAndText(DrawListViewSubItemEventArgs e, bool forbidden)
        {
            Color listBackColor = e.Item.ListView.BackColor;
            Color foreColor;
            Color backColor;

            if (e.Item.Selected && e.Item.ListView.Focused)
            {
                backColor = SystemColors.Highlight;
                foreColor = listBackColor;
            }
            else
            {
                foreColor = forbidden ? SystemColors.GrayText : e.Item.ListView.ForeColor;
                backColor = listBackColor;

                if ((e.Item.Selected && !e.Item.ListView.Focused) || forbidden)
                    backColor = SystemColors.Control;
            }

            e.SubItem.ForeColor = foreColor;
            e.SubItem.BackColor = backColor;

            // Draw the standard header background.
            e.DrawBackground();
        }

        private void listView_DrawColumnHeader(object sender, DrawListViewColumnHeaderEventArgs e)
        {
            e.DrawDefault = true;
        }

        private void listView_DrawItem(object sender, DrawListViewItemEventArgs e)
        {
            e.DrawDefault = false;
        }

        private sealed class MyListView : ListView
        {
            public MyListView()
            {
                DoubleBuffered = true;
            }
        }

        private void listView_Click(object sender, EventArgs e)
        {
            Point mousePos = listView.PointToClient(MousePosition);
            ListViewHitTestInfo hitTest = listView.HitTest(mousePos);
            int columnIndex = hitTest.Item.SubItems.IndexOf(hitTest.SubItem);

            if ((columnIndex != 1) || (m_nodeSelection.Count != 1))
                return;

            var task = CastTag(hitTest.Item.Tag)?.Task;
            if (task == null)
                return;

            if (m_mainForm.SimulationHandler.CanStart && task.Enabled && task.DesignTime)
            {
                task.Execute();
            }
        }

        private void listView_MouseDown(object sender, MouseEventArgs e)
        {
            Point mousePos = listView.PointToClient(MousePosition);
            m_lastHitTest = listView.HitTest(mousePos);
            listView.Invalidate();
        }

        private void listView_MouseUp(object sender, MouseEventArgs e)
        {
            m_lastHitTest = null;
            listView.Invalidate();
        }

        private void listView_MouseLeave(object sender, EventArgs e)
        {
            m_lastHitTest = null;
            listView.Invalidate();
        }

        private void TaskForm_Load(object sender, EventArgs e)
        {
            m_mainForm.SimulationHandler.StateChanged += SimulationHandler_StateChanged;
            RefreshDashboardButton();
        }

        void SimulationHandler_StateChanged(object sender, Core.Execution.MySimulationHandler.StateEventArgs e)
        {
            listView.Invalidate();
        }

        private void TaskForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            m_mainForm.SimulationHandler.StateChanged -= SimulationHandler_StateChanged;
        }

        private void dashboardButton_CheckedChanged(object sender, EventArgs e)
        {
            if (m_nodeSelection.Count != 1)
                return;

            var task = CastTag(listView.SelectedItems[0].Tag).Task;

            if (string.IsNullOrEmpty(task.TaskGroupName))
                m_mainForm.DashboardPropertyToggle(task, EnabledPropertyName, dashboardButton.Checked);
            else
                m_mainForm.DashboardPropertyToggle(task.TaskGroup, task.TaskGroupName, dashboardButton.Checked);
        }

        private void RefreshDashboardButton()
        {
            dashboardButton.Enabled = false;

            if ((listView.SelectedItems.Count != 1) || (m_nodeSelection.Count != 1))
            {
                return;
            }

            var task = CastTag(listView.SelectedItems[0].Tag)?.Task;
            if (task == null)
                return;

            dashboardButton.Enabled = true;
            dashboardButton.Checked = string.IsNullOrEmpty(task.TaskGroupName)
                ? m_mainForm.CheckDashboardContains(task, EnabledPropertyName)
                : m_mainForm.CheckDashboardContains(task.TaskGroup, task.TaskGroupName);
        }
    }
}
