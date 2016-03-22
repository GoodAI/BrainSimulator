using System.Data;
using System.Linq;

namespace World.Tiles
{
    static class TileSetTableParser
    {
        private static readonly DataTable DataTable;

        static TileSetTableParser()
        {
            DataTable = FileHelpers.CsvEngine.CsvToDataTable(@"Module\Tiles\Tilesets\TilesetTable.csv", ';');
        }

        public static int TileNumber(string tileName)
        {
            return int.Parse(
                DataTable.AsEnumerable()
                .First(x => x[0].ToString() == tileName)
                [1]
                .ToString());
        }
    }
}
