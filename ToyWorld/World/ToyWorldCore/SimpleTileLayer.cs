using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Utils;
using VRageMath;
using World.GameActors.Tiles;
using Utils.VRageRIP.Lib.Extensions;

namespace World.ToyWorldCore
{
    public class SimpleTileLayer : ITileLayer
    {
        public SimpleTileLayer(LayerType layerType, int width, int height)
        {
            MyContract.Requires<ArgumentOutOfRangeException>(width > 0, "Tile width has to be positive");
            MyContract.Requires<ArgumentOutOfRangeException>(height > 0, "Tile height has to be positive");
            LayerType = layerType;
            Tiles = ArrayCreator.CreateJaggedArray<Tile[][]>(width, height);
        }

        public Tile[][] Tiles { get; set; }

        public LayerType LayerType { get; set; }

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

        public void GetRectangle(Vector2 pos, Vector2 size, int[] tileTypes)
        {
            /*Vector2I intTopLeft = new Vector2I(pos);
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
                    var tile = Tiles[i][j];
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
            }*/
        }

        public List<Tile> GetAllObjects()
        {
            return Tiles.Cast<Tile>().ToList();
        }
    }
}
