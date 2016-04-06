namespace World.GameActors.Tiles
{
    /// <summary>
    ///     DynamicTile is tile with internal state that can
    /// </summary>
    public abstract class DynamicTile : Tile
    {
        public DynamicTile(ITilesetTable tilesetTable) : base(tilesetTable)
        {
        }
    }
}
