using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.BrainSimulator.Forms
{
    public partial class ValidationForm : DockContent
    {
        internal MyValidator Validator { get; private set; }

        private MainForm m_mainForm;

        public ValidationForm(MainForm mainForm, MyValidator validator)
        {
            InitializeComponent();
            Validator = validator;

            m_mainForm = mainForm;
        }

        public void UpdateListView()
        {
            listView.Items.Clear();
            foreach (MyValidationMessage message in Validator.Messages)
            {
                if (message.Level == MyValidationLevel.ERROR && errorStripButton.Checked ||
                    message.Level == MyValidationLevel.WARNING && warningStripButton.Checked ||
                    message.Level == MyValidationLevel.INFO && infoStripButton.Checked)
                {
                    if (message.Sender != null)
                    {
                        ListViewItem item = new ListViewItem(new string[] { message.Sender.Name, message.Message });
                        item.Tag = message.Sender;
                        item.ImageIndex = (int)message.Level;                        

                        listView.Items.Add(item);
                    }
                    else
                    {
                        ListViewItem item = new ListViewItem(new string[] { "", message.Message });
                        item.ImageIndex = -1;
                        listView.Items.Add(item);
                    }
                }
            }            
        }

        private void ValidationForm_Load(object sender, EventArgs e)
        {
            imageList.Images.Add(Properties.Resources.StatusAnnotations_Information_16xLG_color);
            imageList.Images.Add(Properties.Resources.StatusAnnotations_Warning_16xLG_color);
            imageList.Images.Add(Properties.Resources.Error_red_16x16);
            listView.SmallImageList = imageList;
        }

        private void stripButton_CheckedChanged(object sender, EventArgs e)
        {
            UpdateListView();
        }

        private void listView_MouseDoubleClick(object sender, MouseEventArgs e)
        {
            if (listView.SelectedItems.Count > 0)
            {
                if (listView.SelectedItems[0].Tag is MyNode)
                {
                    MyNode node = listView.SelectedItems[0].Tag as MyNode;

                    if (node.Parent != null)
                    {
                        GraphLayoutForm parentLayoutForm = m_mainForm.OpenGraphLayout(node.Parent);
                        parentLayoutForm.SelectNodeView(node);
                    }
                }
            }
        }
    }
}
