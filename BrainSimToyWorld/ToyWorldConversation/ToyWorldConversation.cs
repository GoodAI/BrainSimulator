using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using GoodAI.BrainSimulator.Forms;
using GoodAI.Core.Execution;
using GoodAI.Core.Nodes;
using GoodAI.ToyWorld;
using GoodAI.ToyWorldAPI;
using WeifenLuo.WinFormsUI.Docking;

namespace ToyWorldConversation
{
    [BrainSimUIExtension]
    public partial class ToyWorldConversation : DockContent
    {
        private readonly MainForm m_mainForm;
        private readonly Font m_boldFont;
        private readonly Font m_normalFont;
        private readonly List<ToyWorldGUI> m_displayNodes;
        private readonly List<ToyWorldGUI> m_interceptNodes;

        private bool m_showDisplay
        {
            get { return checkBox_show_display.Checked; }
        }

        private bool m_showIntercept
        {
            get { return checkBox_show_intercept.Checked; }
        }

        public ToyWorldConversation(MainForm mainForm)
        {
            InitializeComponent();

            m_mainForm = mainForm;
            m_mainForm.SimulationHandler.StateChanged += SimulationHandler_StateChanged;

            m_boldFont = new Font(richTextBox_messages.Font, FontStyle.Bold);
            m_normalFont = richTextBox_messages.Font;

            m_displayNodes = new List<ToyWorldGUI>();
            m_interceptNodes = new List<ToyWorldGUI>();
            FindGuiNodes();
        }

        private void FindGuiNodes()
        {
            m_displayNodes.ForEach(x => x.TextObtained -= NewMessageDisplay);
            m_interceptNodes.ForEach(x => x.TextObtained -= NewMessageIntercept);

            m_displayNodes.Clear();
            m_interceptNodes.Clear();

            foreach (MyNode node in m_mainForm.Project.Network.Children)
                if (node.GetType() == typeof(ToyWorldGUI))
                {
                    ToyWorldGUI guiNode = node as ToyWorldGUI;
                    Debug.Assert(guiNode != null, "guiNode != null");
                    if (guiNode.IsDisplay)
                        m_displayNodes.Add(guiNode);
                    if (guiNode.IsIntercept)
                        m_interceptNodes.Add(guiNode);
                }

            m_displayNodes.ForEach(x => x.TextObtained += NewMessageDisplay);
            m_interceptNodes.ForEach(x => x.TextObtained += NewMessageIntercept);
        }

        private void SimulationHandler_StateChanged(object sender, MySimulationHandler.StateEventArgs e)
        {
            if (e.OldState != MySimulationHandler.SimulationState.STOPPED) return;
            richTextBox_messages.Clear();
            FindGuiNodes();
        }

        private void PrintMessageFrom(string message, string sender)
        {
            richTextBox_messages.SelectionFont = m_boldFont;
            richTextBox_messages.AppendText(sender + "\n");
            richTextBox_messages.SelectionFont = m_normalFont;
            richTextBox_messages.AppendText(message + "\n");
        }

        private void NewMessageDisplay(object sender, MessageEventArgs e)
        {
            if (m_showDisplay)
                Invoke((MethodInvoker)(() =>
                {
                    PrintMessageFrom(e.Message, e.Sender);
                }));
        }

        private void NewMessageIntercept(object sender, MessageEventArgs e)
        {
            if (m_showIntercept)
                Invoke((MethodInvoker)(() =>
                {
                    PrintMessageFrom(e.Message, e.Sender);
                }));
        }

        private void textBox_send_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode != Keys.Enter) return;

            e.SuppressKeyPress = true;  // for disabling the "ding" sound

            m_interceptNodes.ForEach(x => x.Text = textBox_send.Text);

            Invoke((MethodInvoker)(() =>
            {
                PrintMessageFrom(textBox_send.Text, "You");
            }));

            textBox_send.Text = null;
        }
    }
}
