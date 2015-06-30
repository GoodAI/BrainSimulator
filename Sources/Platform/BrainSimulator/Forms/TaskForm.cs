using GoodAI.Core;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Modules.Transforms;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
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
                        ListViewItem item = new ListViewItem(new string[] { task.Name, task.OneShot.ToString() });
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
                task.Enabled = e.Item.Checked;
                
                UpdateTasksEnableState();
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

            if (e.ColumnIndex == 0)
            {
                Point glyphPoint = new Point(4, e.Item.Position.Y + 2);

                MyTask task = e.Item.Tag as MyTask;

                if (string.IsNullOrEmpty(task.TaskGroupName)) 
                {
                    CheckBoxState state = e.Item.Checked ? CheckBoxState.CheckedNormal : CheckBoxState.UncheckedNormal;
                    CheckBoxRenderer.DrawCheckBox(e.Graphics, glyphPoint, state);
                    xOffset = CheckBoxRenderer.GetGlyphSize(e.Graphics, state).Width + 4;    
                }
                else 
                {
                    RadioButtonState state = e.Item.Checked ? RadioButtonState.CheckedNormal : RadioButtonState.UncheckedNormal;
                    RadioButtonRenderer.DrawRadioButton(e.Graphics, glyphPoint, state);
                    xOffset = RadioButtonRenderer.GetGlyphSize(e.Graphics, state).Width + 4;
                }
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
    }
}
