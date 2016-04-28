using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Moq;
using VRageMath;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;
using Xunit;

namespace ToyWorldTests.World
{
    public class AvatarTests
    {
        private Avatar m_avatar;

        public AvatarTests()
        {
            m_avatar = new Avatar("", 0, "Pingu", 0, Vector2.Zero, Vector2.One);
        }

        [Fact]
        public void StartWithoutItem()
        {
            Assert.Equal(null, m_avatar.Tool);
        }

        [Fact]
        public void CanAddToInventory()
        {
            Mock<IPickable> item = new Mock<IPickable>();

            m_avatar.AddToInventory(item.Object);

            Assert.Equal(item.Object, m_avatar.Tool);
        }

        [Fact]
        public void CanHoldOnlyOneItem()
        {
            Mock<IPickable> item1 = new Mock<IPickable>();
            Mock<IPickable> item2 = new Mock<IPickable>();
            m_avatar.AddToInventory(item1.Object);

            // Act
            m_avatar.AddToInventory(item2.Object);

            // Assert
            Assert.Equal(item1.Object, m_avatar.Tool);
        }
    }
}
