using VRageMath;
using World.GameActions;
using World.ToyWorldCore;

namespace World.GameActors.Tiles
{
    /// <summary>
    /// </summary>
    public interface IAutoupdateable
    {
        void Update(IAtlas atlas);

        int NextUpdateAfter { get; }
    }

    /// <summary>
    /// </summary>
    public interface IInteractable
    {
        /// <summary>
        /// Method is called when something apply GameAction on this object.
        /// </summary>
        void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2I position, TilesetTable tilesetTable = null);
    }

    public interface IPickable : IInteractable { }

    public interface ICanPick
    {
        void AddToInventory(IPickable item);
    }
}