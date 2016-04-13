using System.Collections;
using System.Collections.Generic;
using Moq;
using System.Linq;
using World.GameActors.Tiles;
using World.ToyWorldCore;
using Xunit;
using VRageMath;

namespace ToyWorldTests.World
{
    public class SimpleTileLayerTest
    {
        private readonly SimpleTileLayer m_simpleTileLayer;

        private readonly Tile[] m_tileSequenceArray;

        private readonly Tile[][] m_tileArray;

        public SimpleTileLayerTest()
        {
            var tilesetMock1 = new Mock<ITilesetTable>();
            tilesetMock1.Setup(x => x.TileNumber(It.IsAny<string>())).Returns(0);
            var tilesetMock2 = new Mock<ITilesetTable>();
            tilesetMock1.Setup(x => x.TileNumber(It.IsAny<string>())).Returns(1);

            var mockTile1 = new Mock<Tile>(MockBehavior.Default, tilesetMock1.Object);
            var mockTile2 = new Mock<Tile>(MockBehavior.Default, tilesetMock2.Object);
            var tile1 = mockTile1.Object;
            var tile2 = mockTile2.Object;

            Tile[] row0 = new Tile[] { tile1, tile2, tile2 };
            Tile[] row1 = new Tile[] { tile2, tile2, tile1 };
            Tile[] row2 = new Tile[] { tile1, tile2, tile1 };

            m_tileArray = new Tile[][]
                {
                    row0, row1, row2
                };

            m_tileSequenceArray = new Tile[] {
                row0[0], row0[1], row0[2],
                row1[0], row1[1], row1[2],
                row2[0], row2[1], row2[2],
                };

            m_simpleTileLayer = new SimpleTileLayer(LayerType.Obstacle, 3, 3)
            {
                Tiles = m_tileArray
            };
        }

        [Fact]
        public void GetRectangleWholeArray()
        {
            var rectangle = m_simpleTileLayer.GetRectangle(new Rectangle(0,0,3,3));

            var isEqual = rectangle.SequenceEqual(m_tileSequenceArray);

            Assert.True(isEqual);
        }

        [Fact]
        public void GetRectangleSingle()
        {
            var rectangle = m_simpleTileLayer.GetRectangle(new Rectangle(1,1,1,1));

            var isEqual = rectangle.SequenceEqual(new Tile[] { m_tileArray[0][1] });

            Assert.True(isEqual);
        }
    }
}
