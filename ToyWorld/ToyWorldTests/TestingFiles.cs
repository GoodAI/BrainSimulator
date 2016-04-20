using System.IO;
using Utils;

namespace ToyWorldTests
{

    internal static class FileStreams
    {
        public static Stream SmallTmx()
        {
            var s = new FileStream(Globals.TestFileLocation + @"mock_small_stream_world.tmx", FileMode.Open, FileAccess.Read, FileShare.Read);
            return s;
        }

        public static Stream TilesetTableStream()
        {
            var s = new FileStream(Globals.TestFileLocation + @"TilesetTable.csv", FileMode.Open, FileAccess.Read, FileShare.Read);
            return s;
        }

        public static Stream FullTmxStream()
        {
            var path = Globals.TestFileLocation + @"mockup999_pantry_world.tmx";
            return new FileStream(path, FileMode.Open);
        }
    }

}