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
        private IAvatarController m_avatarCtrl;
        private IGameController m_gameCtrl;

        private ToyWorld m_toyWorld
        {
            get { return m_mainForm.Project.World as ToyWorld; }
        }

        private string m_saveFile
        {
            get { return m_toyWorld.SaveFile; }
        }

        private string m_tilesetTable
        {
            get { return m_toyWorld.TilesetTable; }
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
        }

        private void m_mainForm_WorldChanged(object sender, MainForm.WorldChangedEventArgs e)
        {
            if (m_toyWorld == null) return;
            ConnectToToyWorld();
        }

        private void ConnectToToyWorld()
        {
            GameSetup setup = new GameSetup(new FileStream(m_saveFile, FileMode.Open, FileAccess.Read, FileShare.Read), new StreamReader(m_tilesetTable));

            m_gameCtrl = GameFactory.GetThreadSafeGameController(setup);
            m_gameCtrl.Init();
            m_gameCtrl.NewMessage += WorldNewMessage;

            int[] avatarIds = m_gameCtrl.GetAvatarIds();
            if (avatarIds.Length == 0)
            {
                MyLog.ERROR.WriteLine("No avatar found in map!");
                return;
            }

            int myAvatarId = avatarIds[0];
            m_avatarCtrl = m_gameCtrl.GetAvatarController(myAvatarId);
            m_avatarCtrl.NewMessage += AvatarNewMessage;
        }

        private void WorldNewMessage(object sender, MessageEventArgs e)
        {
            if (m_showWorldMessages) richTextBox_messages.AppendText(e.Message);
        }

        private void AvatarNewMessage(object sender, MessageEventArgs e)
        {
            if (m_showAvatarMessages) richTextBox_messages.AppendText(e.Message);
        }

        private void textBox_send_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode != Keys.Enter) return;

            m_avatarCtrl.SendMessage(textBox_send.Text);
        }
    }
}
