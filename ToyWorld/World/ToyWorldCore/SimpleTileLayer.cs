using System;
using System.Diagnostics.Contracts;
using Utils.VRageRIP.Lib.Extensions;
using VRageMath;
using World.GameActors.Tiles;

namespace World.ToyWorldCore
{
    public class SimpleTileLayer : ITileLayer
    {
        private int[] m_tileTypes;
        private readonly int BACKGROUND_TILE_NUMBER = 6;
        private readonly int OBSTACLE_TILE_NUMBER = 7;
        public int Width { get; set; }
        public int Height { get; set; }


        public SimpleTileLayer(LayerType layerType, int width, int height)
        {
            if (width <= 0)
                throw new ArgumentOutOfRangeException("width", "Tile width has to be positive");
            if (height <= 0)
                throw new ArgumentOutOfRangeException("height", "Tile height has to be positive");
            Contract.EndContractBlock();

            m_tileTypes = new int[0];
            LayerType = layerType;
            Height = height;
            Width = width;
            Tiles = ArrayCreator.CreateJaggedArray<Tile[][]>(width, height);
        }

        public Tile[][] Tiles { get; set; }

        public LayerType LayerType { get; set; }

        public int[] GetRectangle(Rectangle rectangle)
        {
            if (m_tileTypes.Length < rectangle.Size.Size())
                m_tileTypes = new int[rectangle.Size.Size()];

            int left = Math.Max(rectangle.Left, Math.Min(0, rectangle.Left + rectangle.Width));
            int right = Math.Min(rectangle.Right, Math.Max(Tiles.Length, rectangle.Right - rectangle.Width));
            // Rectangle origin is in top-left; it's top is thus our bottom
            int bot = Math.Max(rectangle.Top, Math.Min(0, rectangle.Top + rectangle.Height));
            int top = Math.Min(rectangle.Bottom, Math.Max(Tiles[0].Length, rectangle.Bottom - rectangle.Height));


            // TODO : Move to properties
            int idx = 0;
            int defaultTileOffset = 0;
            if (LayerType == LayerType.Background)
            {
                defaultTileOffset = BACKGROUND_TILE_NUMBER;
            }
            else if(LayerType == LayerType.Obstacle)
            {
                defaultTileOffset = OBSTACLE_TILE_NUMBER;
            }

            // Rows before start of map
            for (int j = rectangle.Top; j < bot; j++)
            {
                for (int i = rectangle.Left; i < rectangle.Right; i++)
                    m_tileTypes[idx++] = defaultTileOffset;
            }

            // Rows inside of map
            for (var j = bot; j < top; j++)
            {
                // Tiles before start of map
                for (int i = rectangle.Left; i < left; i++)
                    m_tileTypes[idx++] = defaultTileOffset;

                // Tiles inside of map
                for (var i = left; i < right; i++)
                {
                    var tile = Tiles[i][j];
                    if (tile != null)
                        m_tileTypes[idx++] = tile.TileType;
                    else
                        m_tileTypes[idx++] = 0; // inside map: must be always 0
                }

                // Tiles after end of map
                for (int i = right; i < rectangle.Right; i++)
                    m_tileTypes[idx++] = defaultTileOffset;
            }

            // Rows after end of map
            for (int j = top; j < rectangle.Bottom; j++)
            {
                for (int i = rectangle.Left; i < rectangle.Right; i++)
                    m_tileTypes[idx++] = defaultTileOffset;
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
            if (coordinates.X < 0 || coordinates.Y < 0 || coordinates.X >= Width || coordinates.Y >= Height)
            {
                return new Obstacle(0);
            }
            return Tiles[coordinates.X][coordinates.Y];
        }
    }
}
