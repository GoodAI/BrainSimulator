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
    }
}
