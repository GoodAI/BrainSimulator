using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using VRageMath;
using World.GameActors.Tiles;

namespace World.ToyWorldCore
{
    public class SimpleTileLayer : ITileLayer
    {
        public SimpleTileLayer(LayerType layerType, int width, int height)
        {
            LayerType = layerType;
            Tiles = new Tile[width, height];
        }

        public Tile[,] Tiles { get; set; }

        public LayerType LayerType { get; set; }

        public Tile GetTile(int x, int y)
        {
            return Tiles[x, y];
        }

        public Tile[,] GetRectangle(int x1, int y1, int x2, int y2)
        {
            var xCount = x2 - x1;
            var yCount = y2 - y1;
            var f = new Tile[xCount, yCount];

            for (var i = 0; i < xCount; i++)
            {
                for (var j = 0; j < yCount; j++)
                {
                    f[i, j] = Tiles[x1 + i, y1 + j];
                }
            }

            return f;
        }

        public void GetRectangle(Vector2 pos, Vector2 size, int[] tileTypes)
        {
            Vector2I intTopLeft = new Vector2I(pos);
            Vector2I intBotRight = new Vector2I(pos + size);
            Rectangle rectangle = new Rectangle(intTopLeft, intBotRight - intTopLeft);

            Debug.Assert(tileTypes != null && tileTypes.Length >= rectangle.Size.Size());

            int idx = 0;

            // Rows before start of map
            for (int j = rectangle.Top; j < 0; j++)
            {
                for (int i = rectangle.Left; i < rectangle.Right; i++)
                    tileTypes[idx++] = 0;
            }

            // Rows inside of map
            int xLength = Tiles.GetLength(0);
            int yLength = Tiles.GetLength(1);

            for (var j = 0; j < yLength; j++)
            {
                // Tiles before start of map
                for (int i = rectangle.Left; i < 0; i++)
                    tileTypes[idx++] = 0;

                // Tiles inside of map
                for (var i = 0; i < xLength; i++)
                {
                    var tile = Tiles[i, j];
                    tileTypes[idx++] = tile != null ? tile.TileType : 0;
                }

                // Tiles after end of map
                for (int i = xLength; i < rectangle.Right; i++)
                    tileTypes[idx++] = 0;
            }

            // Rows after end of map
            for (int j = yLength; j < rectangle.Bottom; j++)
            {
                for (int i = rectangle.Left; i < rectangle.Right; i++)
                    tileTypes[idx++] = 0;
            }
        }

        public List<Tile> GetAllObjects()
        {
            return Tiles.Cast<Tile>().ToList();
        }
    }
}
