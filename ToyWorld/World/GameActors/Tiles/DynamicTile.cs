namespace World.GameActors.Tiles
{
    /// <summary>
    ///     DynamicTile is tile with internal state that can
    /// </summary>
    public abstract class DynamicTile : Tile
    {
        public DynamicTile(TilesetTable tilesetTable) : base(tilesetTable)
        {
        }

        public DynamicTile(int tileType) : base(tileType)
        {
        }
    }
}
