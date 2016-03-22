using System.Data;
using System.Linq;

namespace GoodAI.ToyWorldAPI.Tiles
{
    static class TileSetTableParser
    {
        private static DataTable _dataTable;

        static TileSetTableParser()
        {
            _dataTable = FileHelpers.CsvEngine.CsvToDataTable(@"Module\Tiles\Tilesets\TilesetTable.csv", ';');
        }

        public static int TileNumber(string tileName)
        {
            return int.Parse(
                _dataTable.AsEnumerable()
                .First(x => x[0].ToString() == tileName)
                [1]
                .ToString());
        }
    }
}
