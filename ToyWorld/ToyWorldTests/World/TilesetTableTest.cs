using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TmxMapSerializer.Elements;
using World.GameActors.Tiles;
using Xunit;

namespace ToyWorldTests.World
{
    public class TilesetTableTest
    {
        public TilesetTableTest()
        {
            var tilesetTableMemoryStream = FileStreams.GetTilesetTableMemoryStream();
            var tilesetTableStreamReader = new StreamReader(tilesetTableMemoryStream);
            tilesetTableStreamReader.DiscardBufferedData();
            tilesetTableMemoryStream.Position = 0;
            var tilesetTable = new TilesetTable(null, tilesetTableStreamReader);
        }

        [Fact]
        public void CanParse()
        {
            Assert.True(true);
        }
    }
}
