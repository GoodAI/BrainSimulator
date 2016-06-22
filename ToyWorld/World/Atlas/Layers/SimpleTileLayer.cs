using System;
using System.Collections.Generic;
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
        private const int TILESETS_BITS = 12;
        private const int TILESETS_OFFSET = 1 << TILESETS_BITS; // Must be larger than the number of tiles in any tileset and must correspond to the BasicOffset.vert shader
        private const int BACKGROUND_TILE_NUMBER = 6;
        private const int OBSTACLE_TILE_NUMBER = 7;


        #region Summer/winter stuff

        private readonly Random m_random;

        private float m_summer; // Local copy of the Atlas' summer
        private float m_gradient; // Local copy of the Atlas' gradient
        private float m_previousGradient;
        private Vector3 m_summerCache;
        private bool IsWinter { get { return m_summer < 0.25f; } }

        private readonly HashSet<Vector2I> m_snowyTilesSet; // Two values -- present or not -- for summer/winter; Dictionary<> will be needed for more states
        private readonly List<Vector2I> m_snowyTiles;

        private int m_tileCount;
        private int[] m_tileTypes;

        #endregion


        public int Width { get; set; }
        public int Height { get; set; }

        public Tile[][] Tiles { get; set; }
        public bool Render { get; set; }
        public LayerType LayerType { get; set; }


        public SimpleTileLayer(LayerType layerType, int width, int height, Random random = null)
        {
            if (width <= 0)
                throw new ArgumentOutOfRangeException("width", "Tile width has to be positive");
            if (height <= 0)
                throw new ArgumentOutOfRangeException("height", "Tile height has to be positive");
            Contract.EndContractBlock();

            m_random = random ?? new Random();

            m_tileTypes = new int[0];
            LayerType = layerType;
            m_summerCache.Z = m_random.Next();

            Height = height;
            Width = width;

            Tiles = ArrayCreator.CreateJaggedArray<Tile[][]>(width, height);
            m_snowyTilesSet = new HashSet<Vector2I>();
            m_snowyTiles = new List<Vector2I>();

            Render = true;
        }


        public void UpdateTileStates(Atlas atlas)
        {
            m_summer = atlas.Summer;
            m_gradient = atlas.SummerGradient;

            if (m_tileCount == 0) // Nothing to update
                return;

            if (!IsWinter) // This erases any leftover snowy tiles when winter ends
            {
                if (m_snowyTiles.Count > 0)
                {
                    foreach (var snowyTile in m_snowyTiles) // Remove the few leftover snowy tiles directly
                        m_snowyTilesSet.Remove(snowyTile);

                    m_snowyTiles.Clear();
                }
                return;
            }

            if (m_gradient < 0)
                AddSnow(); // It is Oct to Dec
            else
                RemoveSnow(); // It is Jan to Mar

            // TODO: more states defined by atlas

            m_previousGradient = m_gradient;
        }

        private void AddSnow()
        {
            float snowyTilesNeeded = GetSnowyTilesNeededCount();

            if (snowyTilesNeeded <= 0) // We have enough snowy tiles for this part of the year -- we don't need to add any
                return;

            int tileUpdateCount = (int)Math.Ceiling(snowyTilesNeeded); // Always add something if snowyTilesNeeded is not zero

            int repetitions = 0;

            for (int i = 0; i < tileUpdateCount; i++)
            {
                Vector2I position = new Vector2I(m_random.Next(Width), m_random.Next(Height));

                bool added = m_snowyTilesSet.Add(position);

                if (!added && ++repetitions <= 5) // Try more tiles for each missing snowy tile before giving up
                {
                    i--;
                    continue;
                }

                repetitions = 0;
                m_snowyTiles.Add(position);
            }
        }

        private void RemoveSnow()
        {
            if (m_snowyTiles.Count == 0) // No snowy tiles to remove
                return;

            if (m_previousGradient != m_gradient) // Jan has just begun -- snow starts to melt in random order
                m_snowyTiles.ShuffleFisherYates(m_random); // TODO: amortize by shuffling during addition


            float snowyTilesNeeded = GetSnowyTilesNeededCount();

            if (snowyTilesNeeded >= 0) // We don't need to remove any snowy tiles
                return;

            int tileUpdateCount = (int)Math.Ceiling(-snowyTilesNeeded); // Always remove something if snowyTilesNeeded is not zero


            int removeStartIdx = Math.Max(m_snowyTiles.Count - tileUpdateCount, 0);

            for (int i = m_snowyTiles.Count - 1; i >= removeStartIdx; i--)
                m_snowyTilesSet.Remove(m_snowyTiles[i]);

            m_snowyTiles.RemoveEnd(tileUpdateCount);
        }

        private float GetModifiedWinterIntensityFactor(float winterChangeIntensity)
        {
            return (float)Math.Sin(winterChangeIntensity * MathHelper.PiOver2); // Rises faster than y=x (for x in (0,1))
        }

        private float GetSnowyTilesNeededCount()
        {
            float winterIntensityFactor = 1 - m_summer * 4; // It is Oct to Mar, strengthen intensity towards Dec/Jan
            winterIntensityFactor = GetModifiedWinterIntensityFactor(winterIntensityFactor);

            // Update so many snowy tiles, so that the ratio of snowed to non-snowed tiles is exactly winterIntensityFactor
            float currentSnowyTilesRatio = m_snowyTiles.Count / (float)m_tileCount;
            return (winterIntensityFactor - currentSnowyTilesRatio) * m_tileCount; // Negative values for when there are more tiles than the required amount
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
                    {
                        int tileId = tile.TilesetId;

                        if (m_snowyTilesSet.Contains(new Vector2I(i, j)))
                            tileId += TILESETS_OFFSET;

                        m_tileTypes[idx++] = tileId;
                    }
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

        // This is rather slow to compute, but we assume only small portions of grid view will be in sight
        private int GetDefaultTileOffset(int x, int y, int defaultTileOffset)
        {
            if (!IsWinter)
                return defaultTileOffset;

            m_summerCache.X = x;
            m_summerCache.Y = y;
            double hash = (Math.Abs(m_summerCache.GetHash()) % (double)int.MaxValue) / int.MaxValue; // Should be uniformly distributed between 0, 1

            float weatherChangeIntensityFactor = 1 - m_summer * 4; // It is Oct to Mar, use stronger intensity towards Dec/Jan
            weatherChangeIntensityFactor = GetModifiedWinterIntensityFactor(weatherChangeIntensityFactor);

            const float maxWinterIntensityFactor = 0.8f;
            const float winterOffset = 0.02f;
            weatherChangeIntensityFactor *= maxWinterIntensityFactor + winterOffset; // Makes even the hardest winter (summer close to 0) not fill everything with snow
            weatherChangeIntensityFactor -= winterOffset; // Makes winter end a little sooner

            if (hash < weatherChangeIntensityFactor)
                return defaultTileOffset + TILESETS_OFFSET;

            return defaultTileOffset;
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

            if (Tiles[x][y] == null)
                m_tileCount++;

            Tiles[x][y] = tileReplacement;
            return true;
        }

        public bool Add(GameActorPosition gameActorPosition)
        {
            int x = (int)gameActorPosition.Position.X;
            int y = (int)gameActorPosition.Position.Y;

            if (Tiles[x][y] != null)
                return false;

            Tile actor = gameActorPosition.Actor as Tile;
            Tiles[x][y] = actor;

            if (actor != null)
                m_tileCount++;

            return true;
        }

        public void AddInternal(int x, int y, Tile tile)
        {
            Tiles[x][y] = tile;
            m_tileCount++;
        }
    }
}
