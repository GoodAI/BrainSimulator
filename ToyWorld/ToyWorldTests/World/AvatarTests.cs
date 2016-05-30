using System.IO;
using Moq;
using TmxMapSerializer.Elements;
using TmxMapSerializer.Serializer;
using VRageMath;
using World.Atlas.Layers;
using World.GameActors;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;
using World.GameActors.Tiles.ObstacleInteractable;
using World.ToyWorldCore;
using Xunit;

namespace ToyWorldTests.World
{
    public class AvatarTests
    {
        private Avatar m_avatar;
        private IAvatar m_avatarPickuper;
        private ToyWorld m_worldPickupWorld;
        private IAvatar m_eater;

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
            Mock<IPickableGameActor> item = new Mock<IPickableGameActor>();

            m_avatar.AddToInventory(item.Object);

            Assert.Equal(item.Object, m_avatar.Tool);
        }

        [Fact]
        public void CanHoldOnlyOneItem()
        {
            Mock<IPickableGameActor> item1 = new Mock<IPickableGameActor>();
            Mock<IPickableGameActor> item2 = new Mock<IPickableGameActor>();
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
            m_worldPickupWorld = new ToyWorld(map, tilesetTableStreamReader);

            m_avatarPickuper = m_worldPickupWorld.GetAvatar(m_worldPickupWorld.GetAvatarsIds()[0]);
            m_avatarPickuper.PickUp = true;

            Assert.Equal(null, m_avatarPickuper.Tool);

            // Act
            m_avatarPickuper.Update(m_worldPickupWorld.Atlas, It.IsAny<TilesetTable>());

            // Assert
            Assert.IsType<Apple>(m_avatarPickuper.Tool);
            Assert.False(m_avatarPickuper.PickUp);
        }

        [Fact]
        public void AvatarCanLayDown()
        {
            AvatarCanPickUp();

            m_avatarPickuper.PickUp = true;

            // Act
            m_avatarPickuper.Update(m_worldPickupWorld.Atlas, It.IsAny<TilesetTable>());

            // Assert
            Assert.Null(m_avatarPickuper.Tool);
            ILayer<GameActor> obstacleInteractableLayer = m_worldPickupWorld.Atlas.GetLayer(LayerType.ObstacleInteractable);
            Assert.IsType<Apple>(obstacleInteractableLayer.GetActorAt(2, 0));
            Assert.False(m_avatarPickuper.PickUp);
        }

        [Fact]
        public void AvatarLoosingEnergy()
        {
            Stream tmxStream = FileStreams.SmallPickupTmx();
            StreamReader tilesetTableStreamReader = new StreamReader(FileStreams.TilesetTableStream());

            TmxSerializer serializer = new TmxSerializer();
            Map map = serializer.Deserialize(tmxStream);
            m_worldPickupWorld = new ToyWorld(map, tilesetTableStreamReader);

            m_eater = m_worldPickupWorld.GetAvatar(m_worldPickupWorld.GetAvatarsIds()[0]);

            // Act
            for (int i = 0; i < 100; i++)
            {
                m_eater.Update(m_worldPickupWorld.Atlas, It.IsAny<TilesetTable>());
            }

            // Assert
            Assert.True(m_eater.Energy < 1);
        }

        [Fact]
        public void AvatarCanEat()
        {
            AvatarLoosingEnergy();

            m_eater.Interact = true;

            // Act
            m_eater.Update(m_worldPickupWorld.Atlas, It.IsAny<TilesetTable>());

            // Assert
            Assert.Equal(m_eater.Energy, 1);
            ILayer<GameActor> obstacleInteractableLayer = m_worldPickupWorld.Atlas.GetLayer(LayerType.ObstacleInteractable);
            Assert.Null(obstacleInteractableLayer.GetActorAt(2, 0));
        }
    }
}
