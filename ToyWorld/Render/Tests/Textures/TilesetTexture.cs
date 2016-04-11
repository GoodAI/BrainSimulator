using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using Render.RenderObjects.Textures;
using VRageMath;
using Rectangle = System.Drawing.Rectangle;

namespace Render.Tests.Textures
{
    internal class TilesetTexture : TextureBase
    {
        readonly List<TextureBase> m_textures = new List<TextureBase>();

        public int Count { get { return m_textures.Count; } }


        public TilesetTexture(params Stream[] texPath)
        {
            Debug.Assert(texPath != null && texPath.Length > 0 && !texPath.Contains(null));


            foreach (var stream in texPath)
            {
                var bmp = new Bitmap(stream);

                if (bmp.PixelFormat != PixelFormat.Format32bppArgb)
                    throw new ArgumentException("The image on the specified path is not in the required RGBA format.", "texPath");

                BitmapData data = bmp.LockBits(
                    new Rectangle(0, 0, bmp.Width, bmp.Height),
                    ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

                m_textures.Add(new TextureBase(data.Scan0, bmp.Width, bmp.Height));

                bmp.UnlockBits(data);
            }

            Size = m_textures[0].Size;
            Debug.Assert(m_textures.TrueForAll(a => a.Size == Size), "Tilesets have to be of the same dimensionality.");
        }


        public override void Bind()
        {
            Bind(0);
        }

        public void Bind(int tilesetIdx)
        {
            m_textures[tilesetIdx].Bind();
        }
    }
}
