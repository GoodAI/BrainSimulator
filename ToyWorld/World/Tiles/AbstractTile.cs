namespace GoodAI.ToyWorldAPI.Tiles
{
    /// <summary>
    /// All tiles (objects fixed to the grid) are derived from this abstract class.
    /// </summary>
    public abstract class AbstractTile
    {
        /// <summary>
        /// TileType is number in tsx tileset
        /// </summary>
        public abstract int TileType { get; }
    }

    /// <summary>
    /// StaticTile is tile, which cannot be updated, but can be replaced by dynamic tile. Only one static 
    /// </summary>
    public abstract class StaticTile : AbstractTile { }

    /// <summary>
    /// DynamicTile is tile with internal state that can
    /// </summary>
    public abstract class DynamicTile : AbstractTile
    {
        /// <summary>
        /// Method will be called once upon a time according to previus registration
        /// </summary>
        public abstract void Update();

        /// <summary>
        /// 
        /// </summary>
        public abstract void RegisterForUpdate();
    }
}
