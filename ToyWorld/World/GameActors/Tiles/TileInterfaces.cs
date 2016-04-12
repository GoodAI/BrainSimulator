using World.GameActions;
using World.ToyWorldCore;

namespace World.GameActors.Tiles
{
    /// <summary>
    /// </summary>
    public interface IAutoupdateable
    {
        Tile Update(Atlas atlas, TilesetTable tilesetTable, AutoupdateRegister register);
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