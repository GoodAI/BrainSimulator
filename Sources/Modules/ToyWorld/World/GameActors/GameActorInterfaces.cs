using VRageMath;
using World.Atlas;
using World.GameActions;

namespace World.GameActors
{
    /// <summary>
    /// GameActor will be updated in given interval.
    /// </summary>
    public interface IAutoupdateableGameActor : IGameActor
    {
        /// <summary>
        ///
        /// </summary>
        /// <param name="atlas"></param>
        /// <param name="table"></param>
        /// <returns>True if want to be updated again.</returns>
        void Update(IAtlas atlas);

        /// <summary>
        /// In steps. Set 0 for no update.
        /// </summary>
        int NextUpdateAfter { get; }
    }

    /// <summary>
    /// Object which perform some interaction to given GameAction.
    /// </summary>
    public interface IInteractableGameActor : IGameActor
    {
        /// <summary>
        /// Method is called when something apply GameAction on this object.
        /// </summary>
        void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position);
    }

    /// <summary>
    /// Can be picked.
    /// </summary>
    public interface IPickableGameActor : IGameActor
    {
        void PickUp(IAtlas atlas, GameAction gameAction, Vector2 position);
    }

    /// <summary>
    /// For GameActors which are held in hand.
    /// </summary>
    public interface IUsableGameActor : IGameActor
    {
        void Use(GameActorPosition senderPosition, IAtlas atlas);
    }


    public interface ISwitchableGameActor : IGameActor
    {
        ISwitchableGameActor Switch(GameActorPosition gameActorPosition, IAtlas atlas);

        ISwitchableGameActor SwitchOn(GameActorPosition gameActorPosition, IAtlas atlas);

        ISwitchableGameActor SwitchOff(GameActorPosition gameActorPosition, IAtlas atlas);
    }

    public interface ISwitcherGameActor : IGameActor
    {
        void Switch(GameActorPosition gameActorPosition, IAtlas atlas);

        ISwitchableGameActor Switchable { get; set; }
    }

    public interface ICombustibleGameActor : IGameActor
    {
        void Burn(GameActorPosition gameActorPosition, IAtlas atlas);
    }
}
