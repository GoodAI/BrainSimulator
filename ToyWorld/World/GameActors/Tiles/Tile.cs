namespace World.GameActors.Tiles
{
    /// <summary>
    ///     All tiles (objects fixed to the grid) are derived from this abstract class.
    /// </summary>
    public abstract class Tile : GameActor
    {
        protected Tile(ITilesetTable tilesetTable)
        {
            string typeName = GetType().Name;
            TileType = tilesetTable.TileNumber(typeName);
        }

        protected Tile(int tileType)
        {
            TileType = tileType;
        }

        public static explicit operator int(Tile t)
        {
            return t.TileType;
        }

        /// <summary>
        ///     TileType is number in tsx tileset
        /// </summary>
        public readonly int TileType;
    }
}