using System;
using System.Diagnostics.Contracts;
using System.Threading.Tasks;
using VRageMath;
using World.GameActors;
using World.GameActors.Tiles;
using World.Physics;

namespace World.Atlas.Layers
{
    [ContractClass(typeof(TileLayerContracts))]
    public interface ITileLayer : ILayer<Tile>
    {
        int Width { get; set; }

        int Height { get; set; }


        /// <summary>
        /// Updates any internal states of tiles within the layer.
        /// </summary>
        /// <param name="atlas"></param>
        void UpdateTileStates(Atlas atlas);

        /// <summary>
        /// Returns Tiles in given region, where extremes are included.
        /// </summary>
        /// <param name="rectangle"></param>
        /// <param name="tileTypes"></param>
        void GetTileTypesAt(Rectangle rectangle, ushort[] tileTypes);

        /// <summary>
        /// Returns Tiles in given region, where extremes are included.
        /// </summary>
        /// <param name="rectangle"></param>
        /// <param name="tileTypes"></param>
        /// <param name="bufferSize"></param>
        void GetTileTypesAt(Rectangle rectangle, IntPtr tileTypes, int bufferSize);

        /// <summary>
        /// Returns Tiles in given region, where x1 &lt; x2, y1 &lt; y2. x2 and y2 included.
        /// </summary>
        /// <param name="pos"></param>
        /// <param name="size"></param>
        /// <param name="tileTypes"></param>
        void GetTileTypesAt(Vector2I pos, Vector2I size, ushort[] tileTypes);
    }


    [ContractClassFor(typeof(ITileLayer))]
    internal abstract class TileLayerContracts : ITileLayer
    {
        public float Thickness { get; private set; }
        public float SpanIntervalFrom { get; private set; }
        public float SpanIntervalTo { get; private set; }

        public int Width { get; set; }
        public int Height { get; set; }

        public bool Render { get; set; }
        public LayerType LayerType { get; set; }


        public Tile GetActorAt(int x, int y)
        {
            if (x < 0)
                throw new ArgumentOutOfRangeException("x", "x has to be positive");
            if (y < 0)
                throw new ArgumentOutOfRangeException("y", "y has to be positive");
            Contract.EndContractBlock();

            return default(Tile);
        }

        public Tile GetActorAt(Vector2I position)
        {
            return GetActorAt(position.X, position.Y);
        }

        public bool ReplaceWith<TR>(GameActorPosition original, TR replacement)
        {
            return default(bool);
        }

        public bool Add(GameActorPosition gameActorPosition)
        {
            return default(bool);
        }

        public void UpdateTileStates(Atlas atlas)
        { }

        public void GetTileTypesAt(Rectangle rectangle, ushort[] tileTypes)
        {
            if (rectangle.Size.X <= 0 && rectangle.Size.Y <= 0)
                throw new ArgumentOutOfRangeException("rectangle", "values doesn't form a valid rectangle");
            Contract.EndContractBlock();
        }

        public void GetTileTypesAt(Rectangle rectangle, IntPtr tileTypes, int bufferSize)
        {
            if (rectangle.Size.X <= 0 && rectangle.Size.Y <= 0)
                throw new ArgumentOutOfRangeException("rectangle", "values doesn't form a valid rectangle");
            Contract.EndContractBlock();
        }

        public void GetTileTypesAt(Vector2I pos, Vector2I size, ushort[] tileTypes)
        {
            if (size.X <= 0 && size.Y <= 0)
                throw new ArgumentOutOfRangeException("size", "size values doesn't form a valid rectangle");
            Contract.EndContractBlock();
        }
    }
}
