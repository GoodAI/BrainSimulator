using System.Diagnostics;
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

        public void GetRectangle(int x1, int y1, int x2, int y2, int[] tileTypes)
        {
            var xCount = x2 - x1;
            var yCount = y2 - y1;

            Debug.Assert(tileTypes != null && tileTypes.Length >= xCount * yCount);

            for (var j = y1; j < y2; j++)
            {
                if (j < 0)

                for (var i = x1; i < x2; i++)
                {

                    var tile = Tiles[x1 + i, y1 + j];
                    tileTypes[j * xCount + i] = tile != null ? tile.TileType : 0;
                }
            }
        }
    }
}
