using GoodAI.Core.Configuration;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace GoodAI.BrainSimulator.Forms
{    
    public partial class NodeSelectionForm : Form
    {
        private MainForm m_mainForm;

        public NodeSelectionForm(MainForm mainForm)
        {
            m_mainForm = mainForm;
            this.Font = SystemFonts.MessageBoxFont;
            InitializeComponent();
            DialogResult = DialogResult.Cancel;                       
        }

        private void acceptButton_Click(object sender, EventArgs e)
        {            
            FillWithEnabledNodes();
            DialogResult = DialogResult.OK;
            Close();
        }

        private void FillWithEnabledNodes()
        {
            Properties.Settings.Default.ToolBarNodes = new StringCollection();

            foreach(ListViewItem item in nodeListView.Items) 
            {
                if (item.Checked && item.Tag is MyNodeConfig)
                {
                    Properties.Settings.Default.ToolBarNodes.Add((item.Tag as MyNodeConfig).NodeType.Name);
                }
            }            
        }

        private void PopulateNodeListView()
        {
            HashSet<string> enabledNodes = new HashSet<string>();

            if (Properties.Settings.Default.ToolBarNodes != null)
            {
                foreach (string nodeTypeName in Properties.Settings.Default.ToolBarNodes)
                {
                    enabledNodes.Add(nodeTypeName);
                }
            }

            Dictionary<string, List<MyNodeConfig>> knownNodes = new Dictionary<string, List<MyNodeConfig>>();

            foreach (MyNodeConfig nc in MyConfiguration.KnownNodes.Values)
            {
                if (!nc.IsBasicNode && nc.CanBeAdded)
                {
                    if (!knownNodes.ContainsKey(nc.NodeType.Namespace)) 
                    {
                        knownNodes[nc.NodeType.Namespace] = new List<MyNodeConfig>();
                    }
                    knownNodes[nc.NodeType.Namespace].Add(nc);
                }
            }

            knownNodes["Worlds"] = new List<MyNodeConfig>();

            foreach (MyNodeConfig nc in MyConfiguration.KnownWorlds.Values)
            {
                if (!nc.IsBasicNode)
                {
                    knownNodes["Worlds"].Add(nc);
                }
            }

            int i = 0;

            nodeImageList.Images.Add(GeneratePaddedIcon(Properties.Resources.group));
            i++;
            nodeImageList.Images.Add(GeneratePaddedIcon(Properties.Resources.world));
            i++;

            List<string> moduleNameSpaces = knownNodes.Keys.ToList().OrderBy(x => x).ToList();

            foreach (string nameSpace in moduleNameSpaces) 
            {                
                ListViewGroup group = new ListViewGroup(nameSpace, nameSpace.Replace("BrainSimulator.", ""));
                nodeListView.Groups.Add(group);

                // TODO: move
                var categoryItem = new ListViewItem(new string[1] { nameSpace });

                nodeFilterList.Items.Add(categoryItem);

                List<MyNodeConfig> nodesInGroup =  knownNodes[nameSpace].OrderBy(x => x.NodeType.Name).ToList();;
                int row = 0;

                foreach (MyNodeConfig nodeConfig in nodesInGroup) 
                {
                    string author;
                    string status;
                    string summary;

                    bool complete = m_mainForm.Documentation.HasAuthor(nodeConfig.NodeType, out author);
                    complete &= m_mainForm.Documentation.HasStatus(nodeConfig.NodeType, out status);
                    complete &= m_mainForm.Documentation.HasSummary(nodeConfig.NodeType, out summary);

                    MyObsoleteAttribute obsolete = nodeConfig.NodeType.GetCustomAttribute<MyObsoleteAttribute>(true);

                    if (obsolete != null)
                    {
                        status = "Obsolete";
                        summary = "Replaced by: " + MyProject.ShortenNodeTypeName(obsolete.ReplacedBy);
                    }
                    else
                    {
                        if (!complete)
                        {
                            summary = "INCOMPLETE DOCUMENTATION! " + summary;
                        }
                        else
                        {
                            author = author.Replace("&", "&&");
                            status = status.Replace("&", "&&");
                            summary = summary.Replace("&", "&&");                            
                        }
                    }                    

                    ListViewItem item = new ListViewItem(
                        new string[4] { MyProject.ShortenNodeTypeName(nodeConfig.NodeType), author, status, summary });

                    if (row % 2 == 1)
                    {
                        item.BackColor = Color.FromArgb(245, 245, 245);
                    }                    
                    
                    item.Tag = nodeConfig;
                    item.Group = group;
                    item.Checked = enabledNodes.Contains(nodeConfig.NodeType.Name);

                    // forbid to remove currently selected world
                    if (nodeConfig.NodeType == m_mainForm.Project.World.GetType())
                    {
                        item.BackColor = Color.FromArgb(150, 200, 240);  // (light gray-blue)
                        item.ToolTipText = "This world is being used by the current project (can't be deselected).";
                        // NOTE: the item's checkbox can't be disabled, we just override changes in nodeListView_ItemCheck()
                    }

                    if (obsolete != null)
                    {
                        item.ForeColor = SystemColors.GrayText;
                    }
                    else
                    {
                        if (!complete)
                        {
                            item.ForeColor = Color.Red;
                        }
                    }

                    if (nameSpace != "Worlds")
                    {
                        item.ImageIndex = i;
                        nodeImageList.Images.Add(GeneratePaddedIcon(nodeConfig.SmallImage));
                        i++;
                    }
                    else
                    {
                        item.ImageIndex = 1;
                    }

                    nodeListView.Items.Add(item);
                    row++;
                }
            }

            nodeListView.GridLines = true;
        }
        
        private void nodeListView_ItemCheck(object sender, ItemCheckEventArgs e)
        {
            // make sure that the world currently in use stays selected (the list view item's checkbox can't be disabled)
            if ((nodeListView.Items[e.Index].Tag as MyNodeConfig).NodeType == m_mainForm.Project.World.GetType())
            {
                e.NewValue = CheckState.Checked;
            }
        }

        private void NodeSelectionForm_Load(object sender, EventArgs e)
        {                      
            PopulateNodeListView();            
        }

        private Image GeneratePaddedIcon(Image icon)
        {
            Bitmap bmp = new Bitmap(36, 32);
            Graphics g = Graphics.FromImage(bmp);
            g.DrawImage(icon, 2, 0, 32, 32);
            g.Dispose();

            return bmp;
        }        

        private void cancelButton_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void nodeListView_DrawColumnHeader(object sender, DrawListViewColumnHeaderEventArgs e)
        {           
            e.DrawDefault = true;
        }

        private void nodeListView_DrawItem(object sender, DrawListViewItemEventArgs e)
        {
                        
        }

        private static readonly Brush HL_BRUSH = new SolidBrush(SystemColors.Highlight);
        private static readonly Pen LINE_PEN = new Pen(Brushes.LightGray);

        private void nodeListView_DrawSubItem(object sender, DrawListViewSubItemEventArgs e)
        {            
            if (e.ColumnIndex > 0)
            {
                Rectangle bounds = e.SubItem.Bounds;

                if (e.Item.Selected)
                {
                    e.Graphics.FillRectangle(HL_BRUSH, bounds);
                    e.SubItem.ForeColor = SystemColors.HighlightText;
                }
                else
                {
                    e.Graphics.FillRectangle(new SolidBrush(e.Item.BackColor), bounds);
                    e.SubItem.ForeColor = e.Item.ForeColor;
                }                

                e.DrawText(TextFormatFlags.VerticalCenter);                
                e.Graphics.DrawLine(LINE_PEN, bounds.Left, bounds.Top, bounds.Left, bounds.Bottom);                               
            }
            else
            {
                e.DrawDefault = true;       
            }
        }

        private void searchTextBox_TextChanged(object sender, EventArgs e)
        {
            ListViewItem item = nodeListView.FindItemWithText(searchTextBox.Text, true, 0, true);

            if (item != null)
            {
                item.Selected = true;
                item.EnsureVisible();
            }
        }

        public class MyListView : ListView
        {
            public MyListView()
                : base()
            {
                DoubleBuffered = true;
            }
        }

        public class CueTextBox : TextBox
        {
            private static class NativeMethods
            {
                private const uint ECM_FIRST = 0x1500;
                internal const uint EM_SETCUEBANNER = ECM_FIRST + 1;

                [DllImport("user32.dll", CharSet = CharSet.Unicode)]
                public static extern IntPtr SendMessage(IntPtr hWnd, UInt32 Msg, IntPtr wParam, string lParam);
            }

            private string _cue;

            public string Cue
            {
                get
                {
                    return _cue;
                }
                set
                {
                    _cue = value;
                    UpdateCue();
                }
            }

            private void UpdateCue()
            {
                if (IsHandleCreated && _cue != null)
                {
                    NativeMethods.SendMessage(Handle, NativeMethods.EM_SETCUEBANNER, (IntPtr)1, _cue);
                }
            }

            protected override void OnHandleCreated(EventArgs e)
            {
                base.OnHandleCreated(e);
                UpdateCue();
            }
        }
    }
}
