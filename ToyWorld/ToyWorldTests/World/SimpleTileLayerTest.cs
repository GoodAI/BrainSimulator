using System.Collections;
using System.Collections.Generic;
using Moq;
using System.Linq;
using World.GameActors.Tiles;
using World.ToyWorldCore;
using Xunit;

namespace ToyWorldTests.World
{
    public class SimpleTileLayerTest
    {
        private readonly SimpleTileLayer m_simpleTileLayer;

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
            m_simpleTileLayer = new SimpleTileLayer(LayerType.Obstacle, 3, 3)
            {
                Tiles = new Tile[,]
                {
                    {tile1, tile2, tile2},
                    {tile2, tile2, tile1},
                    {tile1, tile2, tile1}
                }
            };
        }

        [Fact]
        public void GetRectangleWholeArray()
        {
            var rectangle = m_simpleTileLayer.GetRectangle(0, 0, 2, 2);
            var origRectangle = m_simpleTileLayer.Tiles;
            var rectangleEnum = rectangle.Cast<Tile>().ToArray();
            var origRectangleEnum = origRectangle.Cast<Tile>().ToArray();

            var isEqual = rectangleEnum.SequenceEqual(origRectangleEnum);

            Assert.True(isEqual);
        }

        [Fact]
        public void GetRectangleSingle()
        {
            var rectangle = m_simpleTileLayer.GetRectangle(1, 1, 1, 1);
            var origRectangle = m_simpleTileLayer.Tiles;
            var rectangleEnum = rectangle.Cast<Tile>().ToArray();
            var origRectangleEnum = origRectangle.Cast<Tile>().ToArray();

            var isEqual = rectangleEnum.SequenceEqual(new Tile[]{origRectangleEnum[4]});

            Assert.True(isEqual);
        }
    }
}
