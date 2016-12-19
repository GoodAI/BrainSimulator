using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using GoodAI.Logging;
using TmxMapSerializer.Elements;
using VRageMath;

namespace World.GameActors.Tiles
{
    public interface ITilesetTable
    {
        /*int TileNumber(string tileName);
        string TileName(int tileNumber);*/
    }

    public class TilesetTable : ITilesetTable
    {
        private readonly Dictionary<string, int> m_namesValuesDictionary;
        private readonly Dictionary<int, string> m_valuesNamesDictionary;


        public readonly List<Tileset> Tilesets = new List<Tileset>();

        public Vector2I TileSize { get; protected set; }
        public Vector2I TileMargins { get; protected set; }
        public Vector2I TileBorder { get; protected set; } // how much the size of the tile is increased
        // correct texture filtering


        public TilesetTable(Map tmxMap, StreamReader tilesetFile)
        {
            if (tmxMap != null)
            {
                Tilesets.AddRange(tmxMap.Tilesets);
                TileSize = new Vector2I(tmxMap.Tilewidth, tmxMap.Tileheight);
                TileMargins = Vector2I.One;
            }

            TileBorder = new Vector2I(TileSize.X, TileSize.Y); // this much border is needed for small resolutions

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

            tilesetFile.Close();

            IEnumerable<DataRow> enumerable = dataTable.Rows.Cast<DataRow>();
            var dataRows = enumerable.ToArray();

            try
            {
                m_namesValuesDictionary = dataRows.Where(x => x["IsDefault"].ToString() == "1")
                    .ToDictionary(x => x["NameOfTile"].ToString(), x => int.Parse(x["PositionInTileset"].ToString()));
            }
            catch (ArgumentException)
            {
                Log.Instance.Error("Duplicate NameOfTiles in tileset table. Try set IsDefault to 0.");
            }

            m_valuesNamesDictionary = dataRows.ToDictionary(x => int.Parse(x["PositionInTileset"].ToString()),
                x => x["NameOfTile"].ToString());


            Assembly assembly = Assembly.GetExecutingAssembly();
            Type[] cachedTypes = assembly.GetTypes().Where(x => x.IsSubclassOf(typeof(Tile))).ToArray();

            Dictionary<string, TilesetIds> tidsDictionary = new Dictionary<string, TilesetIds>();

            foreach (DataRow dataRow in dataRows)
            {
                string className = dataRow["NameOfTile"].ToString();
                string textureName = dataRow["NameOfTexture"].ToString();

                Type type = cachedTypes.FirstOrDefault(x => x.Name == className);
                if (type == null) continue;
                Tile instance = CreateMockInstanceOfTile(type);
                int id = int.Parse(dataRow["PositionInTileset"].ToString());

                if (!tidsDictionary.ContainsKey(className))
                {
                    tidsDictionary[className] = new TilesetIds();
                }
                tidsDictionary[className].Add(id, textureName);

                if (dataRow["IsDefault"].ToString() == "1")
                {
                    instance.DefaultTextureId = id;
                    instance.DefaultTextureName = textureName;
                }

            }

            
            foreach (KeyValuePair<string, TilesetIds> tilesetIds in tidsDictionary)
            {
                Type type = cachedTypes.First(x => x.Name == tilesetIds.Key);
                Tile instance = CreateMockInstanceOfTile(type);
                instance.AlternativeTextures = tilesetIds.Value;
            }

            
        }

        private static Tile CreateMockInstanceOfTile(Type type)
        {
            Tile instance;
            if (type.IsSubclassOf(typeof(DynamicTile)))
            {
                instance = (Tile) Activator.CreateInstance(type, Vector2I.Zero);
            }
            else
            {
                instance = (Tile) Activator.CreateInstance(type);
            }
            return instance;
        }

        /// <summary>
        /// only for mocking
        /// </summary>
        public TilesetTable()
        {

        }


        public IEnumerable<Tileset> GetTilesetImages()
        {
            return Tilesets.Where(t =>
            {
                var fileName = Path.GetFileName(t.Image.Source);
                return fileName != null && fileName.StartsWith("roguelike_");
            });
        }

        public IEnumerable<Tileset> GetOverlayImages()
        {
            return Tilesets.Where(t =>
            {
                var fileName = Path.GetFileName(t.Image.Source);
                return fileName != null && fileName.StartsWith("ui_");
            });
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