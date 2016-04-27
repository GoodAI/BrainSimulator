using System.IO;
using Utils;

namespace ToyWorldTests
{

    internal static class FileStreams
    {
        public static Stream SmallTmx()
        {
            // file copied from World project build output:
            var s = new FileStream(@".\res\Worlds\mock_small_stream_world.tmx", FileMode.Open, FileAccess.Read, FileShare.Read);
            return s;
        }

        public static Stream TilesetTableStream()
        {
            // file copied from World project build output:
            var s = new FileStream(@".\res\GameActors\Tiles\Tilesets\TilesetTable.csv", FileMode.Open, FileAccess.Read, FileShare.Read);
            return s;
        }

        public static Stream FullTmxStream()
        {
            // file copied from World project build output:
            var path = @".\res\Worlds\mockup999_pantry_world.tmx";
            return new FileStream(path, FileMode.Open);
        }
    }

}