using System.Drawing;
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
        private ToyWorldGUI m_guiNode;

        private bool m_showStrings
        {
            get { return checkBox_show_string.Checked; }
        }

        private bool m_showMessages
        {
            get { return checkBox_show_message.Checked; }
        }

        public ToyWorldConversation(MainForm mainForm)
        {
            InitializeComponent();

            m_mainForm = mainForm;
            m_mainForm.SimulationHandler.StateChanged += SimulationHandler_StateChanged;

            m_boldFont = new Font(richTextBox_messages.Font, FontStyle.Bold);
            m_normalFont = richTextBox_messages.Font;
            FindGUINode();
        }

        private void FindGUINode()
        {
            if (m_guiNode != null) return;
            foreach (MyNode node in m_mainForm.Project.Network.Children)
                if (node.GetType() == typeof(ToyWorldGUI))
                {
                    m_guiNode = node as ToyWorldGUI;
                    m_guiNode.MessageObtained += NewMessage;
                    m_guiNode.StringObtained += NewString;
                    return;
                }
        }

        private void SimulationHandler_StateChanged(object sender, MySimulationHandler.StateEventArgs e)
        {
            if (e.OldState != MySimulationHandler.SimulationState.STOPPED) return;
            richTextBox_messages.Clear();
            FindGUINode();
        }

        private void PrintMessageFrom(string message, string sender)
        {
            richTextBox_messages.SelectionFont = m_boldFont;
            richTextBox_messages.AppendText(sender + "\n");
            richTextBox_messages.SelectionFont = m_normalFont;
            richTextBox_messages.AppendText(message + "\n");
        }

        private void NewMessage(object sender, MessageEventArgs e)
        {
            if (m_showMessages)
                Invoke((MethodInvoker)(() =>
                {
                    PrintMessageFrom(e.Message, "Message");
                }));
        }

        private void NewString(object sender, MessageEventArgs e)
        {
            if (m_showStrings)
                Invoke((MethodInvoker)(() =>
                {
                    PrintMessageFrom(e.Message, "String");
                }));
        }

        private void textBox_send_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode != Keys.Enter) return;

            e.SuppressKeyPress = true;  // for disabling the "ding" sound

            m_guiNode.Message = textBox_send.Text;

            Invoke((MethodInvoker)(() =>
            {
                PrintMessageFrom(textBox_send.Text, "You");
            }));

            textBox_send.Text = null;
        }
    }
}
