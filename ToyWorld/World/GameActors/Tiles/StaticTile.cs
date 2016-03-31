using World.GameActions;

namespace World.GameActors.Tiles
{
    /// <summary>
    ///     StaticTile is tile, which cannot be updated, but can be replaced by dynamic tile. Only one static
    /// </summary>
    abstract public class StaticTile : Tile
    {
        protected StaticTile(int tileType) : base()
        {
            TileType = tileType;
        }

        protected StaticTile(GameAction gameAction) : base(gameAction)
        {
        }
    }
}
