using GoodAI.BrainSimulator.Nodes;
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
using GoodAI.BrainSimulator.UserSettings;

namespace GoodAI.BrainSimulator.Forms
{    
    public partial class NodeSelectionForm : Form
    {
        private MainForm m_mainForm;

        private List<UiNodeInfo> m_nodeInfoItems = new List<UiNodeInfo>(100);

        private string m_lastShownFilterName = string.Empty;

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
            AppSettings.SaveSettings(settings =>
            {
                settings.ToolBarNodes = new StringCollection();

                foreach (UiNodeInfo item in m_nodeInfoItems.Where(item => item.ListViewItem.Checked))
                {
                    settings.ToolBarNodes.Add(item.Config.NodeType.Name);
                }
            });
        }

        private void GenerateNodeList()
        {
            var enabledNodes = new HashSet<string>();

            if (Properties.Settings.Default.ToolBarNodes != null)
            {
                foreach (string nodeTypeName in Properties.Settings.Default.ToolBarNodes)
                {
                    enabledNodes.Add(nodeTypeName);
                }
            }

            var knownNodes = new Dictionary<string, List<MyNodeConfig>>();

            foreach (MyNodeConfig nodeConfig in MyConfiguration.KnownNodes.Values)
            {
                if (!nodeConfig.CanBeAdded || (nodeConfig.NodeType == null))
                    continue;

                string nameSpace = nodeConfig.NodeType.Namespace ?? "(unknown)";
                if (!knownNodes.ContainsKey(nameSpace)) 
                {
                    knownNodes[nameSpace] = new List<MyNodeConfig>();
                }

                knownNodes[nameSpace].Add(nodeConfig);
            }

            knownNodes["Worlds"] = new List<MyNodeConfig>();

            foreach (MyWorldConfig nodeConfig in MyConfiguration.KnownWorlds.Values)
            {
                knownNodes["Worlds"].Add(nodeConfig);
            }

            var categorizer = new CategorySortingHat();

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

                List<MyNodeConfig> nodesInGroup = knownNodes[nameSpace].OrderBy(x => x.NodeType.Name).ToList(); ;

                foreach (MyNodeConfig nodeConfig in nodesInGroup) 
                {
                    categorizer.AddNodeAndCategory(nodeConfig);

                    MyObsoleteAttribute obsolete = nodeConfig.NodeType.GetCustomAttribute<MyObsoleteAttribute>(true);

                    bool complete;
                    string[] subitems = ProduceSubitemTexts(nodeConfig, obsolete, out complete);

                    ListViewItem item = new ListViewItem(subitems)
                    {
                        Tag = nodeConfig,
                        Group = @group,
                        Checked = enabledNodes.Contains(nodeConfig.NodeType.Name)
                    };

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

                    m_nodeInfoItems.Add(new UiNodeInfo(item, nodeConfig, string.Join("|", subitems)));
                }
            }

            PopulateCategoryListView(categorizer);
        }

        private string[] ProduceSubitemTexts(MyNodeConfig nodeConfig, MyObsoleteAttribute obsolete, out bool complete)
        {
            string author;
            string status;
            string summary;
            string labels = "";

            complete = m_mainForm.Documentation.HasAuthor(nodeConfig.NodeType, out author);
            complete &= m_mainForm.Documentation.HasStatus(nodeConfig.NodeType, out status);
            complete &= m_mainForm.Documentation.HasSummary(nodeConfig.NodeType, out summary);

            if (obsolete != null)
            {
                status = "Obsolete";
                summary = "Replaced by: " + MyProject.ShortenNodeTypeName(obsolete.ReplacedBy);
            }
            else if (!complete)
            {
                summary = "INCOMPLETE DOCUMENTATION! " + summary;
            }

            if ((nodeConfig.Labels != null) && (nodeConfig.Labels.Count > 0))
            {
                labels = EscapeAmpersands(string.Join(" ", nodeConfig.Labels.Select(label => "#" + label)));
            }

            author = EscapeAmpersands(author);
            status = EscapeAmpersands(status);
            summary = EscapeAmpersands(summary);

            string nodeName = MyProject.ShortenNodeTypeName(nodeConfig.NodeType);

            return new string[] {nodeName, author, status, summary, labels};
        }

        private static string EscapeAmpersands(string text)
        {
            if (text != null)
                text = text.Replace("&", "&&");

            return text;
        }

        private void PopulateCategoryListView(CategorySortingHat categorizer)
        {
            foreach (NodeCategory category in categorizer.SortedCategories)
            {
                filterImageList.Images.Add(GeneratePaddedIcon(category.SmallImage));

                var categoryItem = new ListViewItem(new string[1] { category.Name })
                {
                    Tag = category.Name,  // TODO(Premek): consider tagging with NodeCategory object
                    ImageIndex = filterImageList.Images.Count - 1
                };

                filterList.Items.Add(categoryItem);
            }
        }

        private void PopulateNodeListViewByCategory(string categoryName)
        {
            if (string.IsNullOrEmpty(categoryName))
                return;

            PopulateNodeListView(m_nodeInfoItems
                .Where(item => CategorySortingHat.DetectCategoryName(item.Config) == categoryName)
                .Select(item => item.ListViewItem));
        }

        private void PopulateNodeListViewBySearch(string phrase)
        {
            if (string.IsNullOrEmpty(phrase))
                return;

            string newSearchFilter = "#%#%# search: " + phrase;
            if (m_lastShownFilterName == newSearchFilter)
                return;

            m_lastShownFilterName = newSearchFilter;

            PopulateNodeListView(m_nodeInfoItems
                .Where(item => item.Matches(phrase))
                .Select(item => item.ListViewItem));
        }

        private void PopulateNodeListView(IEnumerable<ListViewItem> items)
        {
            nodeListView.Items.Clear();

            int row = 0;
            bool allSelected = true;

            foreach (var item in items)
            {
                item.BackColor = (row++ % 2 == 1) ? Color.FromArgb(245, 245, 245) : SystemColors.Window;

                nodeListView.Items.Add(item);

                if (!item.Checked)
                    allSelected = false;
            }

            nodeListView.GridLines = true;

            selectAllCheckBox.Enabled = (row > 0);  // avoid multiple enumeration
            selectAllCheckBox.Checked = (row > 0) && allSelected;
        }

        private void ShowNodesFromSelectedCategory()
        {
            if (filterList.SelectedItems.Count == 0)
                return;

            string categoryName = filterList.SelectedItems[0].Tag as string;

            if (categoryName == m_lastShownFilterName)
                return;

            PopulateNodeListViewByCategory(categoryName);
            m_lastShownFilterName = categoryName;
        }

        private void ShowRelevantNodes(bool searchMode)
        {
            const int minSearchCharacters = 2;

            if (searchMode && (searchTextBox.Text.Length >= minSearchCharacters))
            {
                // indicate search is active
                searchTextBox.ForeColor = SystemColors.WindowText;
                searchTextBox.BackColor = SystemColors.Window;
                // TODO(Premek): also unselect category
                
                PopulateNodeListViewBySearch(searchTextBox.Text);
            }
            else  // category mode
            {
                // indicate that search is inactive now
                if (searchMode)  // search, but too few characters
                {
                    searchTextBox.ForeColor = Color.Red;
                    searchTextBox.BackColor = SystemColors.Window;
                }
                else if (searchTextBox.Text.Length >= minSearchCharacters)  // enouch characters, but category mode
                {
                    searchTextBox.ForeColor = SystemColors.InactiveCaptionText;
                    searchTextBox.BackColor = Color.FromArgb(248, 251, 255);
                }
                else  // search too short -- look inviting
                {
                    searchTextBox.ForeColor = Color.Blue;
                    searchTextBox.BackColor = SystemColors.Window;
                }

                ShowNodesFromSelectedCategory();
            }
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
            GenerateNodeList();

            if (filterList.Items.Count > 0)
                filterList.Items[0].Selected = true;
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

        private void nodeFilterList_ItemSelectionChanged(object sender, ListViewItemSelectionChangedEventArgs e)
        {
            ShowRelevantNodes(searchMode: false);
        }

        private void filterList_Enter(object sender, EventArgs e)
        {
            ShowRelevantNodes(searchMode: false);
        }

        private void searchTextBox_TextChanged(object sender, EventArgs e)
        {
            ShowRelevantNodes(searchMode: true);
        }

        private void searchTextBox_Enter(object sender, EventArgs e)
        {
            ShowRelevantNodes(searchMode: true);
        }

        private void NodeSelectionForm_Shown(object sender, EventArgs e)
        {
            searchTextBox.Focus();  // let user type immediately

            AdjustSelectAllCheckBoxPosition();
        }

        private void selectAllCheckBox_CheckedChanged(object sender, EventArgs e)
        {
            foreach (ListViewItem item in nodeListView.Items)
            {
                item.Checked = selectAllCheckBox.Checked;
            }
        }

        private void AdjustSelectAllCheckBoxPosition()
        {
            // a little hack, but the most reliable way I've found
            selectAllCheckBox.Left =
                nodesSplitContainer.Left + nodesSplitContainer.Panel1.Width + nodesSplitContainer.SplitterWidth;
        }

        private void nodesSplitContainer_SplitterMoved(object sender, SplitterEventArgs e)
        {
            AdjustSelectAllCheckBoxPosition();
        }
    }
}
