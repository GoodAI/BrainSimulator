using System;
using System.Diagnostics.Contracts;
using VRageMath;
using World.Physics;

namespace World.GameActors.Tiles
{
    /// <summary>
    ///     All tiles (objects fixed to the grid) are derived from this abstract class.
    /// </summary>
    public abstract class Tile : GameActor
    {

        protected Tile()
        {
            TilesetId = DefaultTextureId;
        }

        protected Tile(int textureId)
        {
            TilesetId = textureId;
        }

        protected Tile(string textureName)
        {
            TilesetId = AlternativeTextures.Id(textureName);
        }

        public void setTexture(string textureName)
        {
            TilesetId = AlternativeTextures.Id(textureName);
        }

        public static explicit operator int(Tile t)
        {
            if (t == null)
                throw new ArgumentNullException("t");
            Contract.EndContractBlock();

            return t.TilesetId;
        }

        public virtual IPhysicalEntity GetPhysicalEntity(Vector2I position)
        {
            return new StaticPhysicalEntity(new RectangleShape(new Vector2(position), Vector2.One));
        }

        public static Vector2 Center(Vector2I position)
        {
            return new Vector2(position) + Vector2.One/2;
        }
    }
}