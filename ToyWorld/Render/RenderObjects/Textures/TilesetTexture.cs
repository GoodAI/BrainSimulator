using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using Utils;

namespace Render.RenderObjects.Textures
{
    internal class TilesetTexture : TextureBase
    {

        readonly List<TextureBase> m_textures = new List<TextureBase>();

        public int Count { get { return m_textures.Count; } }


        public TilesetTexture(params string[] texPath)
        {
            Debug.Assert(texPath != null && texPath.Length > 0);


            foreach (var path in texPath)
            {
                if (path == null)
                {
                    m_textures.Add(null);
                    continue;
                }

                using (FileStream stream = new FileStream(Globals.TestFileLocation + path, FileMode.Open, FileAccess.Read))
                using (Bitmap bmp = new Bitmap(Image.FromStream(stream, true)))
                {
                    if (bmp.PixelFormat != PixelFormat.Format32bppArgb)
                        throw new ArgumentException("The image on the specified path is not in the required RGBA format.", "texPath");

                    BitmapData data = bmp.LockBits(
                        new Rectangle(0, 0, bmp.Width, bmp.Height),
                        ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

                    try
                    {
                        m_textures.Add(new TextureBase(data.Scan0.ArgbToRgbaArray(data.Width * data.Height), bmp.Width, bmp.Height));
                    }
                    finally
                    {
                        bmp.UnlockBits(data);
                    }
                }
            }

            Size = m_textures[0].Size;
            Debug.Assert(m_textures.TrueForAll(a => a.Size == Size), "Tilesets have to be of the same dimensionality.");
        }

        public override void Dispose()
        {
            foreach (var textureBase in m_textures)
                textureBase.Dispose();

            base.Dispose();
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
