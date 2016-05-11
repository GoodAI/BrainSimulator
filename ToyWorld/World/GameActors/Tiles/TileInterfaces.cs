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

        /// <summary>
        /// In steps. Set 0 for no update.
        /// </summary>
        int NextUpdateAfter { get; }
    }

    /// <summary>
    /// </summary>
    public interface IInteractable
    {
        /// <summary>
        /// Method is called when something apply GameAction on this object.
        /// </summary>
        void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position, TilesetTable tilesetTable = null);
    }

    public interface IPickable : IInteractable { }

    public interface ICanPick
    {
        bool AddToInventory(IPickable item);

        IPickable RemoveFromInventory();
    }
}