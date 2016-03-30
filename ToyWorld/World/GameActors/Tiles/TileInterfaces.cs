using World.GameActions;
using World.GameActors.Tiles;

namespace World.Tiles
{
    /// <summary>
    /// </summary>
    public interface IAutoupdateable
    {
        void RegisterForUpdate();

        void Update();
    }

    /// <summary>
    /// </summary>
    public interface INteractable
    {
        /// <summary>
        ///     Method is called when something apply GameAction on it.
        /// </summary>
        Tile ApplyGameAction(GameAction gameAction);
    }
}