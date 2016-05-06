using System;
using System.Diagnostics.Contracts;

namespace World.GameActors.Tiles
{
    /// <summary>
    ///     All tiles (objects fixed to the grid) are derived from this abstract class.
    /// </summary>
    public abstract class Tile : GameActor
    {
        protected Tile(ITilesetTable tilesetTable)
        {
            if (tilesetTable == null)
                throw new ArgumentNullException("tilesetTable");
            Contract.EndContractBlock();

            string typeName = GetType().Name;
            TileType = tilesetTable.TileNumber(typeName);
        }

        protected Tile(int tileType)
        {
            TileType = tileType;
        }

        public static explicit operator int(Tile t)
        {
            if (t == null)
                throw new ArgumentNullException("t");
            Contract.EndContractBlock();

            return t.TileType;
        }

        /// <summary>
        ///     TileType is number in tsx tileset
        /// </summary>
        public readonly int TileType;
    }
}