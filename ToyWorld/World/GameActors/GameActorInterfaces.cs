using VRageMath;
using World.GameActions;
using World.GameActors.Tiles;
using World.ToyWorldCore;


namespace World.GameActors
{
    /// <summary>
    /// GameActor will be updated in given interval.
    /// </summary>
    public interface IAutoupdateableGameActor
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
    public interface IPickable : IGameActor
    {
        void PickUp(IAtlas atlas, GameAction gameAction, Vector2 position, ITilesetTable tilesetTable);
    }

    /// <summary>
    /// For GameActors which are held in hand.
    /// </summary>
    public interface IUsableGameActor
    {
        void Use(GameActorPosition senderPosition, IAtlas atlas, ITilesetTable tilesetTable);
    }


    public interface ISwitchableGameActor
    {
        ISwitchableGameActor Switch(GameActorPosition gameActorPosition, IAtlas atlas, ITilesetTable table);
    }

    public interface ISwitcherGameActor
    {
        void Switch(GameActorPosition gameActorPosition, IAtlas atlas, ITilesetTable table);

        ISwitchableGameActor Switchable { get; set; }
    }

    public interface ICombustibleGameActor
    {
        void Burn(GameActorPosition gameActorPosition, IAtlas atlas, ITilesetTable table);
    }
}
