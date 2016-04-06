namespace World.GameActors.Tiles
{
    /// <summary>
    ///     DynamicTile is tile with internal state that can
    /// </summary>
    public abstract class DynamicTile : Tile
    {
        protected DynamicTile(ITilesetTable tilesetTable) : base(tilesetTable)
        {
        }

        protected DynamicTile(int tileType) : base(tileType)
        {
        }
    }
}
