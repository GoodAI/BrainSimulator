using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using World.GameActors.Tiles;
using Xunit;

namespace ToyWorldTests.World
{
    public class TilesetTableTest
    {
        public TilesetTableTest()
        {
            var tilesetTableMemoryStream = TestingFiles.Files.GetTilesetTableMemoryStream();
            var tilesetTableStreamReader = new StreamReader(tilesetTableMemoryStream);
            tilesetTableStreamReader.DiscardBufferedData();
            tilesetTableMemoryStream.Position = 0;
            var tilesetTable = new TilesetTable(tilesetTableStreamReader);
        }

        [Fact]
        public void CanParse()
        {
            Assert.True(true);
        }
    }
}
