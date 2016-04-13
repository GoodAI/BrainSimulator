using World.GameActors.Tiles;
using Utils.VRageRIP.Lib.Extensions;
using VRageMath;
using System;

namespace World.ToyWorldCore
{
    public class SimpleTileLayer : ITileLayer
    {
        public SimpleTileLayer(LayerType layerType, int width, int height)
        {
            LayerType = layerType;
            Tiles = ArrayCreator.CreateJaggedArray<Tile[][]>(width, height);
        }

        public Tile[][] Tiles { get; set; }

        public LayerType LayerType { get; private set; }

        public Tile GetTile(Vector2I coordinates)
        {
            return Tiles[coordinates.X][coordinates.Y];
        }

        public Tile[] GetRectangle(Rectangle rectangle)
        {
            int totalElementsNumber = rectangle.Height * rectangle.Width;

            var f = new Tile[totalElementsNumber];

            for (int i = 0; i < rectangle.Height; i++)
            {
                Array.Copy(Tiles[rectangle.Top + i], rectangle.Left, f, rectangle.Width * i, rectangle.Width);
            }

            return f;
        }
    }
}
