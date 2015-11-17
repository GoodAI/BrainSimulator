using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.BrainSimulator.Forms
{
    public partial class MemoryBlocksForm : DockContent
    {
        private MainForm m_mainForm;
        private MyNode m_target;
        private bool m_escapeOrEnterPressed;
        private bool m_errorInfoShown = true;

        public MemoryBlocksForm(MainForm mainForm)
        {
            InitializeComponent();
            m_mainForm = mainForm;
        }

        public MyNode Target
        {
            get { return m_target; }
            set
            {
                m_target = value;
                UpdateView();
                if (m_target == null)
                {
                    toolStrip.Enabled = false;
                }
            }
        }

        public void UpdateView() 
        {            
            listView.Items.Clear();
            toolStrip.Enabled = false;
            splitContainer.Panel2Collapsed = true;

            if (Target != null)
            {
                if (m_mainForm.SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED)
                {
                    try
                    {
                        Target.UpdateMemoryBlocks();
                    }
                    catch (Exception e)
                    {
                        MyLog.ERROR.WriteLine("Exeption occured while updating node " + Target.Name +": " + e.Message);
                    }
                }

                List<MyAbstractMemoryBlock> blocks =  MyMemoryManager.Instance.GetBlocks(Target);
                
                
                for (int i = 0; i < Target.InputBranches; i++) 
                {
                    MyAbstractMemoryBlock mb = Target.GetAbstractInput(i);

                    if (mb != null)
                    {
                        if (Target is IMyVariableBranchViewNodeBase)
                        {
                            addListViewItem(mb, "<Input_" + (i + 1) + ">", false);
                        }
                        else if (Target is MyNodeGroup)
                        {
                            addListViewItem(mb, (Target as MyNodeGroup).GroupInputNodes[i].Name, false);
                        }
                        else
                        {
                            addListViewItem(mb, Target.GetInfo().InputBlocks[i].Name, false);
                        }
                    }
                }

                if (Target is MyNodeGroup) 
                {
                    for (int i = 0; i < Target.OutputBranches; i++)
                    {
                        MyAbstractMemoryBlock mb = Target.GetAbstractOutput(i);

                        if (mb != null)
                        {
                            addListViewItem(mb, (Target as MyNodeGroup).GroupOutputNodes[i].Name, false);
                        }
                    }
                }
                else if (Target is MyParentInput) 
                {
                    MyAbstractMemoryBlock mb = Target.GetAbstractOutput(0);

                     if (mb != null)
                     {
                         addListViewItem(mb, MyProject.ShortenMemoryBlockName(mb.Name), false);
                     }
                }

                foreach (MyAbstractMemoryBlock block in blocks)
                {                    
                    addListViewItem(block, block.Name);              
                }
            }
        }

        private void addListViewItem(MyAbstractMemoryBlock block, string name, bool owned = true)
        {
            //SizeT size = block.GetSize();
            //string sizeStr = size > 1024 ? size / 1024 + " KB" : size + " B";

            string typeStr = "";
            if (block.GetType().GetGenericArguments().Length > 0)
            {
                typeStr = block.GetType().GetGenericArguments()[0].Name;
            }

            ListViewItem item = new ListViewItem(new string[] { name, PrintBlockSize(block), typeStr });
            item.Tag = block;

            if (owned)
            {
                item.ForeColor = block.IsOutput ? Color.MidnightBlue : SystemColors.WindowText;
            }
            else
            {
                item.ForeColor = SystemColors.GrayText;
            }
           
            item.UseItemStyleForSubItems = false;
            item.SubItems[1].Font = new System.Drawing.Font(FontFamily.GenericMonospace, 9);

            item.SubItems[1].ForeColor = item.ForeColor;
            item.SubItems[2].ForeColor = item.ForeColor;            

            listView.Items.Add(item);
        }

        private static string PrintBlockSize(MyAbstractMemoryBlock block)
        {
            if (block == null)
                return "?";

            return block.Dims.IsCustom
                ? block.Dims.PrintResult(printTotalSize: true)
                : block.Count.ToString();
        }

        private void listView_SelectedIndexChanged(object sender, EventArgs e)
        {
            toolStrip.Enabled = listView.SelectedItems.Count > 0;

            bool oneItemSelected = (listView.SelectedItems.Count == 1);
            splitContainer.Panel2Collapsed = !oneItemSelected;

            ShowCurrentBlockDimensions();
        }

        private void ShowCurrentBlockDimensions()
        {
            if (listView.SelectedItems.Count < 1)
                return;

            var block = listView.SelectedItems[0].Tag as MyAbstractMemoryBlock;
            if (block != null)
            {
                dimensionsTextBox.Text = block.Dims.PrintResult();
                ShowOrHideErrorInfo("");
            }
        }

        private void addObserverButton_Click(object sender, EventArgs e)
        {
            m_mainForm.CreateAndShowObserverView(listView.SelectedItems[0].Tag as MyAbstractMemoryBlock, Target, typeof(MyMemoryBlockObserver));
        }

        private void addPlotButton_Click(object sender, EventArgs e)
        {
            m_mainForm.CreateAndShowObserverView(listView.SelectedItems[0].Tag as MyAbstractMemoryBlock, Target, typeof(MyTimePlotObserver));
        }

        private void addHostPlotButton_Click(object sender, EventArgs e)
        {
            m_mainForm.CreateAndShowObserverView(listView.SelectedItems[0].Tag as MyAbstractMemoryBlock, Target, typeof(HostTimePlotObserver));
        }

        private void listView_MouseDoubleClick(object sender, MouseEventArgs e)
        {
            addObserverButton_Click(sender, e);
        }

        private void addMatrixObserver_Click(object sender, EventArgs e)
        {
            m_mainForm.CreateAndShowObserverView(listView.SelectedItems[0].Tag as MyAbstractMemoryBlock, Target, typeof(MyMatrixObserver));
        }

        private void addHostMatrixObserver_Click(object sender, EventArgs e)
        {
            m_mainForm.CreateAndShowObserverView(listView.SelectedItems[0].Tag as MyAbstractMemoryBlock, Target, typeof(HostMatrixObserver));
        }

        private void addSpikeObserver_Click(object sender, EventArgs e)
        {
            m_mainForm.CreateAndShowObserverView(listView.SelectedItems[0].Tag as MyAbstractMemoryBlock, Target, typeof(MySpikeRasterObserver));
        }

        private void addHistogramObserver_Click(object sender, EventArgs e)
        {
            m_mainForm.CreateAndShowObserverView(listView.SelectedItems[0].Tag as MyAbstractMemoryBlock, Target, typeof(MyHistogramObserver));
        }

        private void addTextObserver_Click(object sender, EventArgs e)
        {
            m_mainForm.CreateAndShowObserverView(listView.SelectedItems[0].Tag as MyAbstractMemoryBlock, Target, typeof(MyTextObserver));
        }

        private void dimensionsTextBox_TextChanged(object sender, EventArgs e)
        {
            // TODO(P): remove?
        }

        private MyAbstractMemoryBlock TryGetSelectedMemoryBlock()
        {
            if (listView.SelectedItems.Count <= 0)
                return null;

            return listView.SelectedItems[0].Tag as MyAbstractMemoryBlock;
        }

        private void UpdateSizeInfoForSelectedMemoryBlock()
        {
            if (listView.SelectedItems.Count <= 0)
                return;

            listView.SelectedItems[0].SubItems[1].Text =
                PrintBlockSize(listView.SelectedItems[0].Tag as MyAbstractMemoryBlock);
        }

        private bool TrySetMemBlockDimensions()
        {
            var block = TryGetSelectedMemoryBlock();
            if (block == null)
                return false;

            if (string.IsNullOrEmpty(dimensionsTextBox.Text))
                return false;

            try
            {
                block.Dims.Parse(dimensionsTextBox.Text);
            }
            catch (FormatException ex)
            {
                ShowOrHideErrorInfo(ex.Message);  // UX: Even better would be to show a bubble under the textbox.
                return false;
            }

            UpdateSizeInfoForSelectedMemoryBlock();
            ShowOrHideErrorInfo("");
            return true;
        }

        private void ShowOrHideErrorInfo(string info)
        {
            bool showInfo = !string.IsNullOrEmpty(info);

            int infoHeight = errorInfoLabel.Height + 1;

            if (showInfo && !m_errorInfoShown)
            {
                splitContainer.SplitterDistance -= infoHeight;
            }
            else if (!showInfo && m_errorInfoShown)
            {
                splitContainer.SplitterDistance += infoHeight;
            }

            errorInfoLabel.Text = info;
            m_errorInfoShown = showInfo;
        }

        private void dimensionsTextBox_Leave(object sender, EventArgs e)
        {
            if (!m_escapeOrEnterPressed)  // (Esc) => Don't set, (Enter) => Already tried.
            {
                if (!TrySetMemBlockDimensions())
                    return;
            }

            ShowCurrentBlockDimensions();
        }

        private void dimensionsTextBox_KeyPress(object sender, KeyPressEventArgs e)
        {
            m_escapeOrEnterPressed = false;

            if ((e.KeyChar != (char)13) && (e.KeyChar != (char)27))  // only handle Enter and Esc
                return;  

            e.Handled = true;  // prevent "Ding" sound when pressing Enter or Esc
            m_escapeOrEnterPressed = true;

            if (e.KeyChar == (char)13)  // Enter
            {
                if (!TrySetMemBlockDimensions())
                    return;
            }

            listView.Focus();
        }

        private void dimensionsTextBox_Enter(object sender, EventArgs e)
        {
            var block = TryGetSelectedMemoryBlock();
            if (block == null)
                return;

            dimensionsTextBox.Text = block.Dims.PrintSource();
        }
    }
}
