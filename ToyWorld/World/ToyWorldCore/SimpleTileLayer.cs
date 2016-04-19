using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Utils;
using Utils.VRageRIP.Lib.Extensions;
using VRageMath;
using World.GameActors.Tiles;

namespace World.ToyWorldCore
{
    public class SimpleTileLayer : ITileLayer
    {
        private int[] m_tileTypes;


        public SimpleTileLayer(LayerType layerType, int width, int height)
        {
            m_tileTypes = new int[0];
            MyContract.Requires<ArgumentOutOfRangeException>(width > 0, "Tile width has to be positive");
            MyContract.Requires<ArgumentOutOfRangeException>(height > 0, "Tile height has to be positive");
            LayerType = layerType;
            Tiles = ArrayCreator.CreateJaggedArray<Tile[][]>(width, height);
        }

        public Tile[][] Tiles { get; set; }

        public LayerType LayerType { get; set; }

        public int[] GetRectangle(Rectangle rectangle)
        {
            if (m_tileTypes.Length < rectangle.Size.Size())
                m_tileTypes = new int[rectangle.Size.Size()];


            int left = Math.Max(rectangle.Left, 0);
            int right = Math.Min(rectangle.Right, Tiles.Length);
            int top = Math.Max(rectangle.Top, 0);
            int bot = Math.Min(rectangle.Bottom, Tiles[0].Length);

            int idx = 0;

            // Rows before start of map
            for (int j = rectangle.Top; j < top; j++)
            {
                for (int i = rectangle.Left; i < rectangle.Right; i++)
                    m_tileTypes[idx++] = 0;
            }

            // Rows inside of map
            for (var j = top; j < bot; j++)
            {
                // Tiles before start of map
                for (int i = rectangle.Left; i < 0; i++)
                    m_tileTypes[idx++] = 0;

                // Tiles inside of map
                for (var i = left; i < right; i++)
                {
                    var tile = Tiles[i][j];
                    m_tileTypes[idx++] = tile != null ? tile.TileType : 0;
                }

                // Tiles after end of map
                for (int i = right; i < rectangle.Right; i++)
                    m_tileTypes[idx++] = 0;
            }

            // Rows after end of map
            for (int j = bot; j < rectangle.Bottom; j++)
            {
                for (int i = rectangle.Left; i < rectangle.Right; i++)
                    m_tileTypes[idx++] = 0;
            }

            return m_tileTypes;
        }

        public int[] GetRectangle(Vector2I topLeft, Vector2I size)
        {
            Vector2I intBotRight = topLeft + size;
            Rectangle rectangle = new Rectangle(topLeft, intBotRight - topLeft);
            return GetRectangle(rectangle);
        }

        public Tile GetTile(Vector2I coordinates)
        {
            return Tiles[coordinates.X][coordinates.Y];
        }
    }
}
