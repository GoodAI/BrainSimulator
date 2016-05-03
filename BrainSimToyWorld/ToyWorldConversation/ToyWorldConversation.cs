using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using GoodAI.BrainSimulator.Forms;
using GoodAI.Core.Utils;
using GoodAI.ToyWorld;
using GoodAI.ToyWorld.Control;
using GoodAI.ToyWorldAPI;
using ToyWorldFactory;
using WeifenLuo.WinFormsUI.Docking;

namespace ToyWorldConversation
{
    [BrainSimUIExtension]
    public partial class ToyWorldConversation : DockContent
    {
        private readonly MainForm m_mainForm;
        private Font m_boldFont;
        private Font m_normalFont;

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

            m_boldFont = new Font(richTextBox_messages.Font, FontStyle.Bold);
            m_normalFont = richTextBox_messages.Font;
        }

        private void m_mainForm_WorldChanged(object sender, MainForm.WorldChangedEventArgs e)
        {
            if (m_toyWorld == null) return;
            m_toyWorld.WorldInitialized += ConnectToToyWorld;
        }

        private void ConnectToToyWorld(object sender, EventArgs e)
        {
            //m_gameCtrl.Init();
            m_gameCtrl.NewMessage += WorldNewMessage;

            m_avatarCtrl.NewMessage += AvatarNewMessage;
        }

        private void WorldNewMessage(object sender, MessageEventArgs e)
        {
            if (m_showWorldMessages)
                Invoke((MethodInvoker)(() =>
                {
                    richTextBox_messages.SelectionFont = m_boldFont;
                    richTextBox_messages.AppendText("World\n");
                    richTextBox_messages.SelectionFont = m_normalFont;
                    richTextBox_messages.AppendText(e.Message + "\n");
                }));
        }

        private void AvatarNewMessage(object sender, MessageEventArgs e)
        {
            if (m_showAvatarMessages)
                Invoke((MethodInvoker)(() =>
                {
                    richTextBox_messages.SelectionFont = m_boldFont;
                    richTextBox_messages.AppendText("Avatar\n");
                    richTextBox_messages.SelectionFont = m_normalFont;
                    richTextBox_messages.AppendText(e.Message + "\n");
                }));
        }

        private void textBox_send_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode != Keys.Enter) return;

            m_avatarCtrl.SendMessage(textBox_send.Text);
        }
    }
}
