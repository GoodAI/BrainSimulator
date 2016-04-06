using World.GameActors.Tiles;
using World.ToyWorldCore;

namespace World.GameActors.GameObjects
{
    /// <summary>
    /// For GameObjects which are held in hand.
    /// </summary>
    internal interface IUsable
    {
        void Use(Atlas atlas, TilesetTable tilesetTable);
    }
}