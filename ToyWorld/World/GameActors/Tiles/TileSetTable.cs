using System.Collections.Generic;
using System.Data;
using System.Linq;
using FileHelpers;

namespace World.GameActors.Tiles
{
    public class TilesetTable
    {
        private readonly Dictionary<string, int> m_namesValuesDictionary;
        private readonly Dictionary<int, string> m_valuesNamesDictionary;

        public TilesetTable(string filePath)
        {
            var dataTable = CsvEngine.CsvToDataTable(filePath, ';');
            var enumerable = dataTable.Rows.Cast<DataRow>();
            var nameOfTile = dataTable.Columns.Cast<DataColumn>().First(x => x.ColumnName == "NameOfTile").Ordinal;
            var positionInTileset = dataTable.Columns.Cast<DataColumn>().First(x => x.ColumnName == "PositionInTileset").Ordinal;
            m_namesValuesDictionary = enumerable.ToDictionary(x => x[nameOfTile].ToString(), x => int.Parse(x[positionInTileset].ToString()));
            m_valuesNamesDictionary = enumerable.ToDictionary(x => int.Parse(x[positionInTileset].ToString()), x => x[nameOfTile].ToString());
        }

        public int TileNumber(string tileName)
        {
            return m_namesValuesDictionary[tileName];
        }

        public string TileName(int tileNumber)
        {
            return m_valuesNamesDictionary.ContainsKey(tileNumber) ? m_valuesNamesDictionary[tileNumber] : null;
        }
    }
}