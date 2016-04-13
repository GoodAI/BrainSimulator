using VRageMath;
using World.GameActors.Tiles;

namespace World.ToyWorldCore
{
    public interface ITileLayer
    {
        LayerType LayerType { get; set; }

        Tile GetTile(int x, int y);

        /// <summary>
        /// Returns Tiles in given region, where x1 &lt; x2, y1 &lt; y2. x2 and y2 included.
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="y1"></param>
        /// <param name="x2"></param>
        /// <param name="y2"></param>
        /// <returns>Array of array of Tiles in given region.</returns>
        Tile[,] GetRectangle(int x1, int y1, int x2, int y2);

        /// <summary>
        /// Returns Tiles in given region, where x1 &lt; x2, y1 &lt; y2. x2 and y2 included.
        /// </summary>
        /// <param name="size"></param>
        /// <param name="tileTypes"></param>
        /// <param name="pos"></param>
        void GetRectangle(Vector2 pos, Vector2 size, int[] tileTypes);
    }
}
