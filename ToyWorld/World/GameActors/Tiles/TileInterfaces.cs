using World.GameActions;

namespace World.GameActors.Tiles
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
    public interface Interactable
    {
        /// <summary>
        ///     Method is called when something apply GameAction on it.
        /// </summary>
        Tile ApplyGameAction(GameAction gameAction);
    }
}