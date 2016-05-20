using VRageMath;
using World.GameActions;
using World.GameActors.Tiles;
using World.ToyWorldCore;


namespace World.GameActors
{
    /// <summary>
    /// GameActor will be updated in given interval.
    /// </summary>
    public interface IAutoupdateable
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="atlas"></param>
        /// <param name="table"></param>
        /// <returns>True if want to be updated again.</returns>
        void Update(IAtlas atlas, ITilesetTable table);

        /// <summary>
        /// In steps. Set 0 for no update.
        /// </summary>
        int NextUpdateAfter { get; }
    }

    /// <summary>
    /// Object which perform some interaction to given GameAction.
    /// </summary>
    public interface IInteractable : IGameActor
    {
        /// <summary>
        /// Method is called when something apply GameAction on this object.
        /// </summary>
        void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position, ITilesetTable tilesetTable);
    }

    /// <summary>
    /// Can be picked.
    /// </summary>
    public interface IPickable : IInteractable
    {
    }

    /// <summary>
    /// For GameActors which are held in hand.
    /// </summary>
    public interface IUsable
    {
        void Use(GameActorPosition senderPosition, IAtlas atlas, ITilesetTable tilesetTable);
    }
}
