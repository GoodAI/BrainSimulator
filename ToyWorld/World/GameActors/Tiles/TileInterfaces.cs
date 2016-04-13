using World.GameActions;
using World.ToyWorldCore;
using World.WorldInterfaces;

namespace World.GameActors.Tiles
{
    /// <summary>
    /// </summary>
    public interface IAutoupdateable
    {
        Tile Update(IWorld world);
    }

    /// <summary>
    /// </summary>
    public interface IInteractable
    {
        /// <summary>
        /// Method is called when something apply GameAction on this object.
        /// </summary>
        Tile ApplyGameAction(Atlas atlas, GameAction gameAction, TilesetTable tilesetTable);
    }
}