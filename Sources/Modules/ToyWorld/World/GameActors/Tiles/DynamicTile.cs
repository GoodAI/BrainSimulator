using VRageMath;
using World.Atlas.Layers;
using World.Physics;
using World.ToyWorldCore;

namespace World.GameActors.Tiles
{
    public interface IDynamicTile
    {
        Vector2I Position { get; }

        /// <summary>
        /// Serial number of texture in tileset.
        /// </summary>
        int TilesetId { get; set; }

        IPhysicalEntity GetPhysicalEntity(Vector2I position);
    }

    /// <summary>
    ///     DynamicTile is tile with internal state that can
    /// </summary>
    public abstract class DynamicTile : Tile, IDynamicTile
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

        protected GameActorPosition ThisGameActorPosition(LayerType layerType)
        {
            return new GameActorPosition(this, (Vector2) Position, layerType);
        }
    }
}
