using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using Render.RenderObjects.Textures;

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
                var img = Bitmap.FromStream(stream);

            }
            m_textures.AddRange(texPath.Select(a => new TextureBase(null, 0, 0)));
        }

        public static byte[] ReadFully(Stream input)
        {
            byte[] buffer = new byte[16 * 1024];
            using (MemoryStream ms = new MemoryStream())
            {
                int read;
                while ((read = input.Read(buffer, 0, buffer.Length)) > 0)
                {
                    ms.Write(buffer, 0, read);
                }
                return ms.ToArray();
            }
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
