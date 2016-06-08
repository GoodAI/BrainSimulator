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
            m_avatar = new Avatar("", 0, "Pingu", 0, Vector2.Zero, Vector2.One);

            m_avatarController = new AvatarController(m_avatar);
        }

        [Fact]
        public void CanCreateController()
        {
            Assert.NotNull(m_avatarController);
        }

        [Fact]
        public void CanSetPropertiesFrwd()
        {
            IAvatarControls avatarControls = new AvatarControls(0, 0.5f, 0.0f, -0.4f, true, true, true);
            m_avatarController.SetActions(avatarControls);
            Assert.Equal(m_avatar.DesiredSpeed, 0.5f);
            Assert.Equal(m_avatar.DesiredLeftRotation, -0.4f);
            Assert.True(m_avatar.Interact);
            Assert.True(m_avatar.PickUp);
            Assert.True(m_avatar.UseTool);
        }

        [Fact]
        public void CanSetProperties45()
        {
            IAvatarControls avatarControls = new AvatarControls(0, 1f, 1f, -0.4f, true, true, true);
            m_avatarController.SetActions(avatarControls);
            Assert.Equal(m_avatar.DesiredSpeed, 1f);
            Assert.Equal(m_avatar.Direction, -MathHelper.Pi / 4, 2);
            Assert.Equal(m_avatar.DesiredLeftRotation, -0.4f);
            Assert.True(m_avatar.Interact);
            Assert.True(m_avatar.PickUp);
            Assert.True(m_avatar.UseTool);
        }

        [Fact]
        public void CanResetControls()
        {
            IAvatarControls avatarControls = new AvatarControls(0, 0.5f, 0.0f, -0.4f, true, true, true);
            m_avatarController.SetActions(avatarControls);
            Assert.Equal(m_avatar.DesiredSpeed, 0.5f, 2);
            Assert.Equal(m_avatar.DesiredLeftRotation, -0.4f);
            Assert.True(m_avatar.Interact);
            Assert.True(m_avatar.PickUp);
            Assert.True(m_avatar.UseTool);
            m_avatarController.ResetControls();
            Assert.Equal(m_avatar.DesiredSpeed, 0.0f);
            Assert.Equal(m_avatar.DesiredLeftRotation, 0.0f);
            Assert.True(!m_avatar.Interact);
            Assert.True(!m_avatar.PickUp);
            Assert.True(!m_avatar.UseTool);
        }
    }
}
