using VRageMath;
using World.GameActions;
using World.GameActors.GameObjects;
using World.ToyWorldCore;

namespace World.GameActors.Tiles
{
    /// <summary>
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
    /// </summary>
    public interface IInteractable : IGameActor
    {
        /// <summary>
        /// Method is called when something apply GameAction on this object.
        /// </summary>
        void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position, ITilesetTable tilesetTable);
    }

    public interface IPickable : IInteractable { }

    public interface ICanPick
    {
        bool AddToInventory(IPickable item);

        IPickable RemoveFromInventory();
    }

    public interface ITileDetector
    {
        /// <summary>
        /// If this is true, object is not detected until center of object is within tile.
        /// </summary>
        bool RequiresCenterOfObject { get; }

        void ObjectDetected(IGameObject gameObject, IAtlas atlas, ITilesetTable tilesetTable);
    }

    public interface IHeatSource : IDynamicTile
    {
        float Heat { get; }
    }

    public interface ISwitchable
    {
        GameActor Switch(IAtlas atlas, ITilesetTable table);
    }

    public interface ISwitcher
    {
        void Switch(IAtlas atlas, ITilesetTable table);

        ISwitchable Switchable { get; set; }
    }
}