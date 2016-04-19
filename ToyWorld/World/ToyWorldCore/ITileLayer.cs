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
        Tile GetTile(Vector2I coordinates);

        /// <summary>
        /// Returns Tiles in given region, where extremes are included.
        /// </summary>
        /// <param name="rectangle"></param>
        int[] GetRectangle(Rectangle rectangle);

        /// <summary>
        /// Returns Tiles in given region, where x1 &lt; x2, y1 &lt; y2. x2 and y2 included.
        /// </summary>
        /// <param name="size"></param>
        /// <param name="pos"></param>
        int[] GetRectangle(Vector2I pos, Vector2I size);
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

        public int[] GetRectangle(Rectangle rectangle)
        {
            MyContract.Requires<ArgumentOutOfRangeException>(rectangle.Size.X > 0 && rectangle.Size.Y > 0, "Y values doesn't form a valid rectangle");

            return default(int[]);
        }

        public int[] GetRectangle(Vector2I pos, Vector2I size)
        {
            MyContract.Requires<ArgumentOutOfRangeException>(size.X > 0 && size.Y > 0, "Y values doesn't form a valid rectangle");

            return default(int[]);
        }

        public LayerType LayerType { get; set; }

        public Tile GetTile(int x, int y)
        {
            return default(Tile);
        }

        public Tile GetTile(Vector2I coordinates)
        {
            return default(Tile);
        }
    }
}
