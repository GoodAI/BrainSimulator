using System.Drawing;
using Render.RenderObjects.Textures;

namespace Render.Tests.Textures
{
    internal class TilesetTexture : TextureBase
    {
        public TilesetTexture(string texPath)
            : base(new Bitmap(texPath))
        { }
    }
}
