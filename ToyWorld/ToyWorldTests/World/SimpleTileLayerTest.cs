using System.Collections;
using System.Collections.Generic;
using Moq;
using System.Linq;
using World.GameActors.Tiles;
using World.ToyWorldCore;
using Xunit;
using VRageMath;
using World.Atlas.Layers;

namespace ToyWorldTests.World
{
    public class SimpleTileLayerTest
    {
        private readonly SimpleTileLayer m_simpleTileLayer;

        private readonly Tile[] m_tileSequenceArray;

        private readonly Tile[][] m_tileArray;

        public SimpleTileLayerTest()
        {
            var tilesetMock0 = new Mock<ITilesetTable>();
            tilesetMock0.Setup(x => x.TileNumber(It.IsAny<string>())).Returns(0);
            var tilesetMock1 = new Mock<ITilesetTable>();
            tilesetMock1.Setup(x => x.TileNumber(It.IsAny<string>())).Returns(1);

            var mockTile1 = new Mock<Tile>(MockBehavior.Default, tilesetMock0.Object);
            var mockTile2 = new Mock<Tile>(MockBehavior.Default, tilesetMock1.Object);
            var tile0 = mockTile1.Object;
            var tile1 = mockTile2.Object;

            Tile[] row0 = new Tile[] { tile0, tile1, tile1 };
            Tile[] row1 = new Tile[] { tile1, tile1, tile0 };
            Tile[] row2 = new Tile[] { tile0, tile1, tile0 };

            m_tileArray = new[]
                {
                    row0, row1, row2
                };

            m_tileSequenceArray = new Tile[] {
                row0[0], row1[0], row2[0],
                row0[1], row1[1], row2[1],
                row0[2], row1[2], row2[2],
                };

            m_simpleTileLayer = new SimpleTileLayer(LayerType.Obstacle, 3, 3)
            {
                Tiles = m_tileArray
            };
        }

        [Fact]
        public void GetRectangleWholeArray()
        {
            Rectangle rectangle = new Rectangle(0, 0, 3, 3);
            int[] tileTypes = new int[rectangle.Size.Size()];
            m_simpleTileLayer.GetTileTypesAt(rectangle, tileTypes);

            var isEqual = tileTypes.SequenceEqual(m_tileSequenceArray.Select(x => x.TilesetId));

            Assert.True(isEqual);
        }

        [Fact]
        public void GetRectangleSingle()
        {
            Rectangle rectangle = new Rectangle(1, 1, 1, 1);
            int[] tileTypes = new int[rectangle.Size.Size()];
            m_simpleTileLayer.GetTileTypesAt(rectangle, tileTypes);

            var isEqual = tileTypes.SequenceEqual(new[] { (int)m_tileArray[1][1] });

            Assert.True(isEqual);
        }
    }
}
