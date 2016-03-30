using System.Collections.Generic;
using System.Data;
using System.Linq;
using FileHelpers;

namespace World.Tiles
{
    public class TileSetTableParser
    {
        private readonly Dictionary<string, int> m_namesValuesDictionary;

        public TileSetTableParser(string filePath = @"GameActors\Tiles\Tilesets\TilesetTable.csv")
        {
            m_namesValuesDictionary = ParseTable(filePath);
        }

        public int TileNumber(string tileName)
        {
            return m_namesValuesDictionary[tileName];
        }

        private Dictionary<string, int> ParseTable(string filePath)
        {
            var dataTable = CsvEngine.CsvToDataTable(filePath, ';');
            var enumerable = dataTable.Rows.Cast<DataRow>();
            return enumerable.ToDictionary(x => x[0].ToString(), x => int.Parse(x[1].ToString()));
        }
    }
}