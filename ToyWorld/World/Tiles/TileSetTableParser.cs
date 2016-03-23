using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Linq;

namespace World.Tiles
{
    internal static class TileSetTableParser
    {
        private static Dictionary<string, int> m_namesValuesDictionary;
        private static string _filePath = @"Tiles\Tilesets\TilesetTable.csv";

        public static int TileNumber(string tileName)
        {
            if (m_namesValuesDictionary == null)
            {
                DataTable dataTable = FileHelpers.CsvEngine.CsvToDataTable(_filePath, ';');
                IEnumerable<DataRow> enumerable = dataTable.Rows.Cast<DataRow>();
                m_namesValuesDictionary = enumerable.ToDictionary(x => x[0].ToString(), x => int.Parse(x[1].ToString()));
            }
            return m_namesValuesDictionary[tileName];
        }

        public static void ChangeSourceFile(string filePath)
        {
            _filePath = filePath;
        }
    }
}
