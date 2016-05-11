using VRageMath;

namespace World.GameActors.Tiles
{
    /// <summary>
    ///     DynamicTile is tile with internal state that can
    /// </summary>
    public abstract class DynamicTile : Tile
    {
        public Vector2I Position { get; private set; }

        protected DynamicTile(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable)
        {
            Position = position;
        }

        protected DynamicTile(int tileType, Vector2I position)
            : base(tileType)
        {
            Position = position;
        }
    }
}
