using World.GameActions;

namespace World.GameActors.Tiles
{
    /// <summary>
    ///     All tiles (objects fixed to the grid) are derived from this abstract class.
    /// </summary>
    public abstract class Tile : GameActor
    {
        private int m_tileType = int.MinValue;

        protected Tile(TilesetTable tilesetTable)
        {
            string TypeName = GetType().Name;
            m_tileType = tilesetTable.TileNumber(TypeName);
        }

        protected Tile(int tileType)
        {
            m_tileType = tileType;
        }

        /// <summary>
        ///     TileType is number in tsx tileset
        /// </summary>
        public int TileType {
            get { return m_tileType; }
            protected set { m_tileType = value; }
        }
    }
}