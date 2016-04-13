using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.IO;
using System.Linq;
using TmxMapSerializer.Elements;
using VRageMath;

namespace World.GameActors.Tiles
{
    public interface ITilesetTable
    {
        int TileNumber(string tileName);
        string TileName(int tileNumber);
    }

    public class TilesetTable : ITilesetTable
    {
        private readonly Dictionary<string, int> m_namesValuesDictionary;
        private readonly Dictionary<int, string> m_valuesNamesDictionary;


        public readonly List<Tileset> Tilesets = new List<Tileset>();

        public Vector2I TileSize { get; protected set; }
        public Vector2I TileMargins { get; protected set; }


        public TilesetTable(Map tmxMap, StreamReader tilesetFile)
        {
            if (tmxMap != null)
            {
                Tilesets.AddRange(tmxMap.Tileset);
                TileSize = new Vector2I(tmxMap.Tilewidth, tmxMap.Tileheight);
                // TODO: deserialize margins
                TileMargins = Vector2I.One;
            }

            var dataTable = new DataTable();
            string readLine = tilesetFile.ReadLine();

            Debug.Assert(readLine != null, "readLine != null");
            foreach (string header in readLine.Split(';'))
            {
                dataTable.Columns.Add(header);
            }


            while (!tilesetFile.EndOfStream)
            {
                string line = tilesetFile.ReadLine();
                Debug.Assert(line != null, "line != null");
                string[] row = line.Split(';');
                if (dataTable.Columns.Count != row.Length)
                    break;
                DataRow newRow = dataTable.NewRow();
                foreach (DataColumn column in dataTable.Columns)
                {
                    newRow[column.ColumnName] = row[column.Ordinal];
                }
                dataTable.Rows.Add(newRow);
            }

            IEnumerable<DataRow> enumerable = dataTable.Rows.Cast<DataRow>();
            var dataRows = enumerable.ToArray();

            m_namesValuesDictionary = dataRows.Where(x => x["IsDefault"].ToString() == "1")
                .ToDictionary(x => x["NameOfTile"].ToString(), x => int.Parse(x["PositionInTileset"].ToString()));
            m_valuesNamesDictionary = dataRows.ToDictionary(x => int.Parse(x["PositionInTileset"].ToString()), x => x["NameOfTile"].ToString());
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