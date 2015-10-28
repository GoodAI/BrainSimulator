﻿using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Drawing;
using System.Reflection;
using System.Windows.Forms;
using System.Windows.Forms.VisualStyles;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.BrainSimulator.Forms
{
    public partial class TaskForm : DockContent
    {
        private MainForm m_mainForm;
        private MyWorkingNode m_target;
        private bool isUpdating;

        private int lastSelectedTaskIndex = -1;

        public MyWorkingNode Target
        {
            get { return m_target; }
            set 
            {
                if (Target != null && value != null && value.GetType() == Target.GetType() && listView.SelectedIndices.Count == 1)
                {
                    lastSelectedTaskIndex = listView.SelectedIndices[0];
                }
                else if (value != null)
                {
                    lastSelectedTaskIndex = 0;
                }
                else
                {
                    lastSelectedTaskIndex = -1;
                }

                m_target = value;                
                UpdateTaskView();
            }
        }

        private void UpdateTaskView() 
        {
            isUpdating = true;

            listView.Items.Clear();
            if (m_target != null)
            {
                foreach (PropertyInfo taskPropInfo in m_target.GetInfo().TaskOrder)
                {
                    MyTask task = m_target.GetTaskByPropertyName(taskPropInfo.Name);

                    if (task != null)
                    {
                        ListViewItem item = new ListViewItem(new string[] { task.Name, task.OneShot ? "Init" : "" });
                        item.Checked = task.Enabled;
                        item.Tag = task;                        
                        listView.Items.Add(item);
                    }
                }               
            }

            isUpdating = false;

            if (lastSelectedTaskIndex != -1 && listView.Items.Count > lastSelectedTaskIndex)
            {
                listView.Items[lastSelectedTaskIndex].Selected = true;
                listView.Invalidate();
            }
            else
            {
                m_mainForm.TaskPropertyView.Target = null;
            }
        }

        private void UpdateTasksEnableState()
        {
            isUpdating = true;

            foreach (ListViewItem item in listView.Items)
            {
                MyTask task = item.Tag as MyTask;
                item.Checked = task.Enabled;           
            }

            isUpdating = false;
        }

        public TaskForm(MainForm mainForm)
        {
            m_mainForm = mainForm;
            
            InitializeComponent();
        }

        private void listView_ItemChecked(object sender, ItemCheckedEventArgs e)
        {
            if (!isUpdating)
            {
                MyTask task = e.Item.Tag as MyTask;

                if (!task.DesignTime)
                {
                    task.Enabled = e.Item.Checked;
                    UpdateTasksEnableState();
                }
            }            
        }

        private void listView_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (listView.SelectedItems.Count == 1)
            {
                m_mainForm.TaskPropertyView.Target = listView.SelectedItems[0].Tag as MyTask;
            }
        }

        private static readonly Brush HL_BRUSH = new SolidBrush(SystemColors.Highlight);
        private static readonly Pen LINE_PEN = new Pen(Brushes.LightGray);

        private void listView_DrawSubItem(object sender, DrawListViewSubItemEventArgs e)
        {            
            Rectangle bounds = e.SubItem.Bounds;

            if (e.ColumnIndex == 0)
            {
                bounds.Width = bounds.X + e.Item.SubItems[1].Bounds.X;
            }

            //toggle colors if the item is highlighted 
            if (e.Item.Selected && e.Item.ListView.Focused)
            {
                e.SubItem.BackColor = SystemColors.Highlight;
                e.SubItem.ForeColor = e.Item.ListView.BackColor;
            }
            else if (e.Item.Selected && !e.Item.ListView.Focused)
            {
                e.SubItem.BackColor = SystemColors.Control;
                e.SubItem.ForeColor = e.Item.ListView.ForeColor;
            }
            else
            {
                e.SubItem.BackColor = e.Item.ListView.BackColor;
                e.SubItem.ForeColor = e.Item.ListView.ForeColor;
            }

            // Draw the standard header background.
            e.DrawBackground();

            int xOffset = 0;

            MyTask task = e.Item.Tag as MyTask;
            MyTaskInfoAttribute taskAttrib = task.GetType().GetCustomAttribute<MyTaskInfoAttribute>(false);            

            if (e.ColumnIndex == 0)
            {
                Point glyphPoint = new Point(4, e.Item.Position.Y + 2);            

                if (!string.IsNullOrEmpty(task.TaskGroupName)) 
                {
                    RadioButtonState state = e.Item.Checked ? RadioButtonState.CheckedNormal : RadioButtonState.UncheckedNormal;
                    RadioButtonRenderer.DrawRadioButton(e.Graphics, glyphPoint, state);
                    xOffset = RadioButtonRenderer.GetGlyphSize(e.Graphics, state).Width + 4;                    
                }
                else if (task.DesignTime)
                {                                        
                    xOffset = CheckBoxRenderer.GetGlyphSize(e.Graphics, CheckBoxState.UncheckedNormal).Width + 4;
                }
                else
                {
                    CheckBoxState state = e.Item.Checked ? CheckBoxState.CheckedNormal : CheckBoxState.UncheckedNormal;
                    CheckBoxRenderer.DrawCheckBox(e.Graphics, glyphPoint, state);
                    xOffset = CheckBoxRenderer.GetGlyphSize(e.Graphics, state).Width + 4;    
                }
            }
            else if (e.ColumnIndex == 1 && task.DesignTime)
            {                
                PushButtonState buttonState = PushButtonState.Disabled;

                if (m_mainForm.SimulationHandler.CanStart && task.Enabled)
                {                    
                    buttonState =
                        m_lastHitTest != null && e.Item == m_lastHitTest.Item && e.SubItem == m_lastHitTest.SubItem ?
                        PushButtonState.Pressed : PushButtonState.Normal;
                }

                ButtonRenderer.DrawButton(e.Graphics, e.Bounds, "Execute", listView.Font, false, buttonState);
            }
            
            //add a 2 pixel buffer the match default behavior
            Rectangle rec = new Rectangle(e.Bounds.X + 2 + xOffset, e.Bounds.Y + 2, e.Bounds.Width - 4, e.Bounds.Height - 4);

            //TODO  Confirm combination of TextFormatFlags.EndEllipsis and TextFormatFlags.ExpandTabs works on all systems.  MSDN claims they're exclusive but on Win7-64 they work.
            TextFormatFlags flags = TextFormatFlags.Left | TextFormatFlags.EndEllipsis | TextFormatFlags.ExpandTabs | TextFormatFlags.SingleLine;

            //If a different tabstop than the default is needed, will have to p/invoke DrawTextEx from win32.
            TextRenderer.DrawText(e.Graphics, e.SubItem.Text, e.Item.ListView.Font, rec, e.SubItem.ForeColor, flags);         
        }

        private void listView_DrawColumnHeader(object sender, DrawListViewColumnHeaderEventArgs e)
        {
            e.DrawDefault = true;
        }

        private void listView_DrawItem(object sender, DrawListViewItemEventArgs e)
        {
            e.DrawDefault = false;
        }

        public class MyListView : ListView
        {
            public MyListView()
                : base()
            {
                DoubleBuffered = true;
            }
        }

        private void listView_Click(object sender, EventArgs e)
        {
            Point mousePos = listView.PointToClient(Control.MousePosition);
            ListViewHitTestInfo hitTest = listView.HitTest(mousePos);
            int columnIndex = hitTest.Item.SubItems.IndexOf(hitTest.SubItem);

            if (columnIndex == 1)
            {
                MyTask task = hitTest.Item.Tag as MyTask;
                MyTaskInfoAttribute taskAttrib = task.GetType().GetCustomAttribute<MyTaskInfoAttribute>(false);

                if (m_mainForm.SimulationHandler.CanStart && task.Enabled && taskAttrib != null && taskAttrib.DesignTime)
                {
                    task.Execute();
                }
            }
        }

        private ListViewHitTestInfo m_lastHitTest;

        private void listView_MouseDown(object sender, MouseEventArgs e)
        {
            Point mousePos = listView.PointToClient(Control.MousePosition);
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
        }

        void SimulationHandler_StateChanged(object sender, Core.Execution.MySimulationHandler.StateEventArgs e)
        {
            listView.Invalidate();
        }

        private void TaskForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            m_mainForm.SimulationHandler.StateChanged -= SimulationHandler_StateChanged;
        }
    }
}
