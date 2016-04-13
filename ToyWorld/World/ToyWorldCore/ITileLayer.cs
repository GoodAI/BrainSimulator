using System;
using System.Diagnostics.Contracts;
using Utils;
using VRageMath;
using World.GameActors.Tiles;
using World.ToyWorldCore;

namespace World.ToyWorldCore
{
    [ContractClass(typeof(TileLayerContracts))]
    public interface ITileLayer : ILayer<Tile>
    {
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


    [ContractClassFor(typeof(ITileLayer))]
    internal abstract class TileLayerContracts : ITileLayer
    {
        public Tile[,] GetRectangle(int x1, int y1, int x2, int y2)
        {
            MyContract.Requires<ArgumentOutOfRangeException>((x2 - x1 + 1) > 0, "X values doesn't form a valid rectangle");
            MyContract.Requires<ArgumentOutOfRangeException>((y2 - y1 + 1) > 0, "Y values doesn't form a valid rectangle");

            return default(Tile[,]);
        }

        public void GetRectangle(Vector2 pos, Vector2 size, int[] tileTypes) { }

        public LayerType LayerType { get; set; }

        public Tile GetTile(int x, int y)
        {
            return default(Tile);
        }
    }
}
