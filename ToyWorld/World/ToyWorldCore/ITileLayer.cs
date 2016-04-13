using VRageMath;
using World.GameActors.Tiles;

namespace World.ToyWorldCore
{
    public interface ITileLayer
    {
        LayerType LayerType { get; }

        Tile GetTile(Vector2I coordinates);
        
        /// <summary>
        /// Returns Tiles in given region.
        /// </summary>
        /// <param name="rectangle"></param>
        /// <returns>Array of Tiles in given region.</returns>
        Tile[] GetRectangle(Rectangle rectangle);
    }
}