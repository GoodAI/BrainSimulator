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
            var dataRows = enumerable as DataRow[] ?? enumerable.ToArray();

            var nameOfTile = dataTable.Columns.Cast<DataColumn>().First(x => x.ColumnName == "NameOfTile").Ordinal;
            var positionInTileset = dataTable.Columns.Cast<DataColumn>().First(x => x.ColumnName == "PositionInTileset").Ordinal;
            var isDefault = dataTable.Columns.Cast<DataColumn>().First(x => x.ColumnName == "IsDefault").Ordinal;
            
            m_namesValuesDictionary = dataRows.Where(x => x[isDefault].ToString() == "1")
                .ToDictionary(x => x[nameOfTile].ToString(), x => int.Parse(x[positionInTileset].ToString()));
            m_valuesNamesDictionary = dataRows.ToDictionary(x => int.Parse(x[positionInTileset].ToString()), x => x[nameOfTile].ToString());
        }

        /// <summary>
        /// only for mocking
        /// </summary>
        public TilesetTable()
        {
            
        }

        public virtual int TileNumber(string tileName)
        {
            return m_namesValuesDictionary[tileName];
        }

        public virtual string TileName(int tileNumber)
        {
            return m_valuesNamesDictionary.ContainsKey(tileNumber) ? m_valuesNamesDictionary[tileNumber] : null;
        }
    }
}