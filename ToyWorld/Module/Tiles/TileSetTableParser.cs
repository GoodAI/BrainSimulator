using System.Data;
using System.Linq;

namespace GoodAI.ToyWorldAPI.Tiles
{
    class TileSetTableParser
    {
        private DataTable _dataTable;
        public TileSetTableParser()
            : this(@"Module\Tiles\Tilesets\TilesetTable.csv") { }

        public TileSetTableParser(string filePath)
        {
            _dataTable = FileHelpers.CsvEngine.CsvToDataTable(filePath, ';');
        }

        public int TileNumber(string tileName)
        {
            return int.Parse(
                _dataTable.AsEnumerable()
                .First(x => x[0].ToString() == tileName)
                [1]
                .ToString());
        }
    }
}
