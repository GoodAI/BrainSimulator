using System;
using System.Drawing;
using System.Windows.Forms;
using GoodAI.BrainSimulator.Forms;
using GoodAI.Core.Execution;
using GoodAI.ToyWorld;
using GoodAI.ToyWorld.Control;
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

        private IAvatarController m_avatarCtrl
        {
            get { return m_toyWorld.AvatarCtrl; }
        }

        private IGameController m_gameCtrl
        {
            get { return m_toyWorld.GameCtrl; }
        }

        private ToyWorld m_toyWorld
        {
            get { return m_mainForm.Project.World as ToyWorld; }
        }

        private bool m_showAvatarMessages
        {
            get { return checkBox_show_agent.Checked; }
        }

        private bool m_showWorldMessages
        {
            get { return checkBox_show_world.Checked; }
        }

        public ToyWorldConversation(MainForm mainForm)
        {
            InitializeComponent();

            m_mainForm = mainForm;
            m_mainForm.WorldChanged += m_mainForm_WorldChanged;
            m_mainForm.SimulationHandler.StateChanged += SimulationHandler_StateChanged;

            m_boldFont = new Font(richTextBox_messages.Font, FontStyle.Bold);
            m_normalFont = richTextBox_messages.Font;
        }

        private void SimulationHandler_StateChanged(object sender, MySimulationHandler.StateEventArgs e)
        {
            if (e.OldState == MySimulationHandler.SimulationState.STOPPED)
                richTextBox_messages.Clear();
        }

        private void m_mainForm_WorldChanged(object sender, MainForm.WorldChangedEventArgs e)
        {
            if (m_toyWorld == null) return;
            m_toyWorld.WorldInitialized += ConnectToToyWorld;
        }

        private void ConnectToToyWorld(object sender, EventArgs e)
        {
            if (m_gameCtrl == null || m_avatarCtrl == null) return;
            m_gameCtrl.NewMessage += WorldNewMessage;
            m_avatarCtrl.NewMessage += AvatarNewMessage;
        }

        private void PrintMessageFrom(string message, string sender)
        {
            richTextBox_messages.SelectionFont = m_boldFont;
            richTextBox_messages.AppendText(sender + "\n");
            richTextBox_messages.SelectionFont = m_normalFont;
            richTextBox_messages.AppendText(message + "\n");
        }

        private void WorldNewMessage(object sender, MessageEventArgs e)
        {
            if (m_showWorldMessages)
                Invoke((MethodInvoker)(() =>
                {
                    PrintMessageFrom(e.Message, "World");
                }));
        }

        private void AvatarNewMessage(object sender, MessageEventArgs e)
        {
            if (m_showAvatarMessages)
                Invoke((MethodInvoker)(() =>
                {
                    PrintMessageFrom(e.Message, "Avatar");
                }));
        }

        private void textBox_send_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode != Keys.Enter || m_avatarCtrl == null) return;

            e.SuppressKeyPress = true;  // for disabling the "ding" sound
            m_avatarCtrl.MessageIn = textBox_send.Text;
            Invoke((MethodInvoker)(() =>
            {
                PrintMessageFrom(textBox_send.Text, "You");
            }));

            textBox_send.Text = null;
        }
    }
}
