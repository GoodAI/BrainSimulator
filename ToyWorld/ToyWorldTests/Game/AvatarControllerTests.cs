using Game;
using GoodAI.ToyWorld.Control;
using VRageMath;
using World.GameActors.GameObjects;
using Xunit;

namespace ToyWorldTests.Game
{
    public class AvatarControllerTests
    {
        private IAvatarController m_avatarController;
        private IAvatar m_avatar;

        public AvatarControllerTests()
        {
            m_avatar = new Avatar("Pingu", 0, Vector2.Zero, Vector2.One);

            m_avatarController = new AvatarController(m_avatar);
        }

        [Fact]
        public void CanCreateController()
        {
            Assert.NotNull(m_avatarController);
        }

        [Fact]
        public void CanSetProperties()
        {
            IAvatarControls avatarControls = new AvatarControls(0, 0.5f, -0.4f, true, true, true);
            m_avatarController.SetActions(avatarControls);
            Assert.True(m_avatar.DesiredSpeed == 0.5f);
            Assert.True(m_avatar.DesiredRotation == -0.4f);
            Assert.True(m_avatar.Interact);
            Assert.True(m_avatar.PickUp);
            Assert.True(m_avatar.Use);
        }

        [Fact]
        public void CanResetControls()
        {
            IAvatarControls avatarControls = new AvatarControls(0, 0.5f, -0.4f, true, true, true);
            m_avatarController.SetActions(avatarControls);
            Assert.True(m_avatar.DesiredSpeed == 0.5f);
            Assert.True(m_avatar.DesiredRotation == -0.4f);
            Assert.True(m_avatar.Interact);
            Assert.True(m_avatar.PickUp);
            Assert.True(m_avatar.Use);
            m_avatarController.ResetControls();
            Assert.True(m_avatar.DesiredSpeed == 0.0f);
            Assert.True(m_avatar.DesiredRotation == 0.0f);
            Assert.True(!m_avatar.Interact);
            Assert.True(!m_avatar.PickUp);
            Assert.True(!m_avatar.Use);
        }
    }
}
