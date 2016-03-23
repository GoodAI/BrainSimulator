using World.GameActions;

namespace World.Tiles
{
    /// <summary>
    /// 
    /// </summary>
    public interface Autoupdateable
    {
        void RegisterForUpdate();

        void Update();
    }

    /// <summary>
    /// 
    /// </summary>
    public interface Interactable
    {
        /// <summary>
        /// Method is called when something apply GameAction on it.
        /// </summary>
        AbstractTile ApplyGameAction(GameAction gameAction);
    }
}