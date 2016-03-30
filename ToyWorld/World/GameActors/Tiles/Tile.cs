namespace World.GameActors.Tiles
{
    /// <summary>
    ///     All tiles (objects fixed to the grid) are derived from this abstract class.
    /// </summary>
    public abstract class Tile : GameActor
    {
        private int m_tileType = int.MinValue;

        /// <summary>
        ///     TileType is number in tsx tileset
        /// </summary>
        public int TileType {
            get { return m_tileType; }
            protected set { m_tileType = value; }
        }
    }
}