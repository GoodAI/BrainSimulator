using System.Runtime.InteropServices;
using VRageMath;
using World.GameActors.GameObjects;
using World.Physics;
using Xunit;

namespace ToyWorldTests.Physics
{
    public class BasicAvatarMoverTest
    {
        private Avatar m_avatar;
        private BasicAvatarMover m_basicAvatarMover;

        public BasicAvatarMoverTest()
        {
            m_avatar = new Avatar("", 0, "Pingu", 0, Vector2.Zero, Vector2.One);
            m_basicAvatarMover = new BasicAvatarMover();
            BasicAvatarMover basicAvatarMover = m_basicAvatarMover;
        }

        [Fact]
        public void CanSetPhysicalProperies()
        {
            m_avatar.DesiredRotation = 1;
            m_avatar.DesiredSpeed = 1;
            m_basicAvatarMover.SetAvatarMotion(m_avatar);
            Assert.True(m_avatar.PhysicalEntity.ForwardSpeed > 0f);
            Assert.True(m_avatar.PhysicalEntity.RotationSpeed > 0f);
        }
    }
}
