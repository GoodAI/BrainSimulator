using System.IO;
using Moq;
using TmxMapSerializer.Elements;
using TmxMapSerializer.Serializer;
using VRageMath;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;
using World.ToyWorldCore;
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

        [Fact]
        public void AvatarCanPickUp()
        {
            Stream tmxStream = FileStreams.SmallPickupTmx();
            StreamReader tilesetTableStreamReader = new StreamReader(FileStreams.TilesetTableStream());

            TmxSerializer serializer = new TmxSerializer();
            Map map = serializer.Deserialize(tmxStream);
            ToyWorld world = new ToyWorld(map, tilesetTableStreamReader);

            IAvatar avatar = world.GetAvatar(world.GetAvatarsIds()[0]);
            avatar.PickUp = true;

            Assert.Equal(null, avatar.Tool);

            // Act
            avatar.Update(world.Atlas);

            // Assert
            Assert.IsType<Apple>(avatar.Tool);
        }
    }
}
