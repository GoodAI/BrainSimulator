using System.IO;
using GoodAI.BrainSimulator.Forms;
using GoodAI.Core.Utils;
using GoodAI.ToyWorld;
using GoodAI.ToyWorld.Control;
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

        public ToyWorldConversation(MainForm mainForm)
        {
            InitializeComponent();

            m_mainForm = mainForm;
            m_mainForm.WorldChanged += m_mainForm_WorldChanged;
        }

        void m_mainForm_WorldChanged(object sender, MainForm.WorldChangedEventArgs e)
        {
            if (m_toyWorld == null) return;
            ConnectToToyWorld();
        }

        private void ConnectToToyWorld()
        {
            GameSetup setup = new GameSetup(new FileStream(m_saveFile, FileMode.Open, FileAccess.Read, FileShare.Read), new StreamReader(m_tilesetTable));

            m_gameCtrl = GameFactory.GetThreadSafeGameController(setup);
            m_gameCtrl.Init();

            int[] avatarIds = m_gameCtrl.GetAvatarIds();
            if (avatarIds.Length == 0)
            {
                MyLog.ERROR.WriteLine("No avatar found in map!");
                return;
            }

            int myAvatarId = avatarIds[0];
            m_avatarCtrl = m_gameCtrl.GetAvatarController(myAvatarId);
        }
    }
}
