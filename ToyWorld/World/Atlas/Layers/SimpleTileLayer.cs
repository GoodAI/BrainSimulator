using System;
using System.Diagnostics.Contracts;
using Utils.VRageRIP.Lib.Extensions;
using VRageMath;
using World.GameActors;
using World.GameActors.Tiles;
using World.GameActors.Tiles.Obstacle;
using World.Physics;

namespace World.Atlas.Layers
{
    public class SimpleTileLayer : ITileLayer
    {
        private const int TILESETS_OFFSET = 2 << 12; // Must be larger than the number of tiles in any tileset
        private const int BACKGROUND_TILE_NUMBER = 6;
        private const int OBSTACLE_TILE_NUMBER = 7;

        private readonly Random m_random;

        private Vector3 m_summerCache; // Z value holds the Atlas' summer state

        private int m_tileCount;
        private int[] m_tileTypes;

        public int Width { get; set; }
        public int Height { get; set; }


        public Tile[][] Tiles { get; set; }

        public byte[][] TileStates { get; set; }

        public bool Render { get; set; }

        public LayerType LayerType { get; set; }


        public SimpleTileLayer(LayerType layerType, int width, int height)
        {
            if (width <= 0)
                throw new ArgumentOutOfRangeException("width", "Tile width has to be positive");
            if (height <= 0)
                throw new ArgumentOutOfRangeException("height", "Tile height has to be positive");
            Contract.EndContractBlock();

            m_random = new Random();
            m_tileTypes = new int[0];
            LayerType = layerType;
            Height = height;
            Width = width;
            Tiles = ArrayCreator.CreateJaggedArray<Tile[][]>(width, height);
            TileStates = ArrayCreator.CreateJaggedArray<byte[][]>(width, height);
            Render = true;
        }

        public void UpdateTileStates(float summer)
        {
            m_summerCache.Z = summer;

            const float tileUpdateCountFactor = 0.02f;
            float summerDepthFactor = summer + 0.5f;
            int tileUpdateCount = (int)(m_tileCount * tileUpdateCountFactor * summerDepthFactor);

            for (int i = 0; i < tileUpdateCount; i++)
            {
                int x = m_random.Next(Width);
                int y = m_random.Next(Height);

                if (summer - 0.5f < 0)
                    TileStates[x][y] = 0; // summer
                else
                    TileStates[x][y] = 1; // winter

                // TODO: more states defined by atlas
            }
        }


        public Tile GetActorAt(int x, int y)
        {
            if (x < 0 || y < 0 || x >= Width || y >= Height)
            {
                return new Obstacle(0);
            }
            return Tiles[x][y];
        }

        public Tile GetActorAt(Shape shape)
        {
            Vector2I position = new Vector2I(Vector2.Floor(shape.Position));
            return Tiles[position.X][position.Y];
        }

        public Tile GetActorAt(Vector2I coordinates)
        {
            return GetActorAt(coordinates.X, coordinates.Y);
        }

        public int[] GetRectangle(Vector2I topLeft, Vector2I size)
        {
            Vector2I intBotRight = topLeft + size;
            Rectangle rectangle = new Rectangle(topLeft, intBotRight - topLeft);
            return GetRectangle(rectangle);
        }

        public int[] GetRectangle(Rectangle rectangle)
        {
            if (m_tileTypes.Length < rectangle.Size.Size())
                m_tileTypes = new int[rectangle.Size.Size()];

            // Use cached getter value
            int viewRight = rectangle.Right;

            int left = Math.Max(rectangle.Left, Math.Min(0, rectangle.Left + rectangle.Width));
            int right = Math.Min(viewRight, Math.Max(Tiles.Length, viewRight - rectangle.Width));
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
            else if (LayerType == LayerType.Obstacle)
            {
                defaultTileOffset = OBSTACLE_TILE_NUMBER;
            }

            // Rows before start of map
            for (int j = rectangle.Top; j < bot; j++)
            {
                for (int i = rectangle.Left; i < viewRight; i++)
                    m_tileTypes[idx++] = GetDefaultTileOffset(i, j, defaultTileOffset);
            }

            // Rows inside of map
            for (var j = bot; j < top; j++)
            {
                // Tiles before start of map
                for (int i = rectangle.Left; i < left; i++)
                    m_tileTypes[idx++] = GetDefaultTileOffset(i, j, defaultTileOffset);

                // Tiles inside of map
                for (var i = left; i < right; i++)
                {
                    var tile = Tiles[i][j];
                    if (tile != null)
                        m_tileTypes[idx++] = tile.TilesetId + TileStates[i][j] * TILESETS_OFFSET;
                    else
                        m_tileTypes[idx++] = 0; // inside map: must be always 0
                }

                // Tiles after end of map
                for (int i = right; i < viewRight; i++)
                    m_tileTypes[idx++] = GetDefaultTileOffset(i, j, defaultTileOffset);
            }

            // Rows after end of map
            for (int j = top; j < rectangle.Bottom; j++)
            {
                for (int i = rectangle.Left; i < viewRight; i++)
                    m_tileTypes[idx++] = GetDefaultTileOffset(i, j, defaultTileOffset);
            }

            return m_tileTypes;
        }

        private int GetDefaultTileOffset(int x, int y, int defaultTileOffset)
        {
            m_summerCache.X = x;
            m_summerCache.Y = y;
            double hash = m_summerCache.GetHash() / (double)int.MaxValue; // Should be uniformly distributed between 0, 1

            if (hash <= m_summerCache.Z)
                return defaultTileOffset;

            return defaultTileOffset + TileStates[x][y] * TILESETS_OFFSET;
        }

        public bool ReplaceWith<T>(GameActorPosition original, T replacement)
        {
            int x = (int)Math.Floor(original.Position.X);
            int y = (int)Math.Floor(original.Position.Y);
            Tile item = GetActorAt(x, y);

            if (item != original.Actor) return false;

            Tiles[x][y] = null;
            Tile tileReplacement = replacement as Tile;

            if (replacement == null)
            {
                m_tileCount--;
                return true;
            }

            Tiles[x][y] = tileReplacement;
            return true;
        }

        public bool Add(GameActorPosition gameActorPosition)
        {
            int x = (int)gameActorPosition.Position.X;
            int y = (int)gameActorPosition.Position.Y;
            if (Tiles[x][y] != null)
            {
                return false;
            }
            Tiles[x][y] = gameActorPosition.Actor as Tile;
            m_tileCount++;
            return true;
        }
    }
}
