using World.GameActions;
using World.ToyWorldCore;

namespace World.GameActors.Tiles
{
    /// <summary>
    /// </summary>
    public interface IAutoupdateable
    {
        void RegisterForUpdate();

        Tile Update(Atlas atlas, TilesetTable tilesetTable);
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