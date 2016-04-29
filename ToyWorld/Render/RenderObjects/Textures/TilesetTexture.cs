using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using OpenTK.Graphics.OpenGL;

namespace Render.RenderObjects.Textures
{
    internal class TilesetTexture : TextureBase
    {
        readonly List<TextureBase> m_textures = new List<TextureBase>();

        public int Count { get { return m_textures.Count; } }


        public TilesetTexture(params TilesetImage[] tilesetImages)
        {
            Debug.Assert(tilesetImages != null && tilesetImages.Length > 0);


            foreach (TilesetImage tilesetImage in tilesetImages)
            {
                if (tilesetImage == null || tilesetImage.ImagePath == null)
                {
                    m_textures.Add(null);
                    continue;
                }
                using (FileStream stream = new FileStream(tilesetImage.ImagePath, FileMode.Open, FileAccess.Read))
                using (Bitmap bmp = new Bitmap(Image.FromStream(stream, true)))
                {
                    if (bmp.PixelFormat != System.Drawing.Imaging.PixelFormat.Format32bppArgb)
                        throw new ArgumentException("The image on the specified path is not in the required RGBA format.", "texPath");

                    int tilesPerRow = bmp.Width / (tilesetImage.TileSize.X + tilesetImage.TileMargin.X);
                    int tilesPerColumn = bmp.Height / (tilesetImage.TileSize.Y + tilesetImage.TileMargin.Y);

                    Bitmap bmpTextureWithBorders = new Bitmap(
                        tilesPerRow * (tilesetImage.TileSize.X + tilesetImage.TileMargin.X + tilesetImage.TileBorder.X * 2),
                        tilesPerColumn * (tilesetImage.TileSize.Y + tilesetImage.TileMargin.Y + tilesetImage.TileBorder.Y * 2),
                        System.Drawing.Imaging.PixelFormat.Format32bppArgb);

                    BitmapData dataOrig = bmp.LockBits(
                        new Rectangle(0, 0, bmp.Width, bmp.Height),
                        ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format32bppArgb);

                    BitmapData dataNew = bmpTextureWithBorders.LockBits(
                        new Rectangle(0, 0, bmpTextureWithBorders.Width, bmpTextureWithBorders.Height),
                        ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format32bppArgb);

                    IncreaseTileBorders(dataOrig, dataNew, tilesPerRow, tilesPerColumn,
                        tilesetImage);

                    try
                    {
                        m_textures.Add(
                            new TextureBase(
                                dataNew.Scan0.ArgbToRgbaArray(dataNew.Width * dataNew.Height),
                                bmpTextureWithBorders.Width, bmpTextureWithBorders.Height,
                                minFilter: TextureMinFilter.LinearMipmapLinear,
                                magFilter: TextureMagFilter.Linear,
                                generateMipmap: true));
                    }
                    finally
                    {
                        bmp.UnlockBits(dataOrig);
                        bmpTextureWithBorders.UnlockBits(dataNew);
                        bmpTextureWithBorders.Dispose();
                    }
                }
            }

            Size = m_textures[0].Size;
            Debug.Assert(m_textures.TrueForAll(a => a.Size == Size), "Tilesets have to be of the same dimensionality.");
        }



        // simulates OpenGL's GL_CLAMP_TO_EDGE on the tileset
        protected static void IncreaseTileBorders(BitmapData dataOrig, BitmapData dataNew, int tilesPerRow, int tilesPerColumn,
            TilesetImage tilesetImage)
        {
            VRageMath.Vector2I tileSize = tilesetImage.TileSize;
            VRageMath.Vector2I tileBorder = tilesetImage.TileBorder;
            VRageMath.Vector2I tileMargin = tilesetImage.TileMargin;


            // Get the address of the first line.
            IntPtr dataOrigPtr = dataOrig.Scan0;
            IntPtr dataNewPtr = dataNew.Scan0;

            // Declare an array to hold the bytes of the bitmap.
            int bytesOrig = Math.Abs(dataOrig.Stride) * dataOrig.Height;
            int bytesNew = Math.Abs(dataNew.Stride) * dataNew.Height;
            byte[] dataBytesOrig = new byte[bytesOrig];
            byte[] dataBytesNew = new byte[bytesNew];

            // Copy the RGB values into the array.
            System.Runtime.InteropServices.Marshal.Copy(dataOrigPtr, dataBytesOrig, 0, bytesOrig);

            int iRowOrig = 0;
            int iRowNew = 0;

            for (int i = 0; i < tilesPerColumn; i++)
            {

                // space between tiles
                if (i != 0)
                {
                    iRowOrig += tileMargin.Y;
                    iRowNew += tileMargin.Y;
                }

                // tiles
                for (int iTileY = 0; iTileY < tileSize.Y; iTileY++)
                {
                    int iColumnOrig = 0;
                    int iColumnNew = 0;

                    // the copy operations multiply bytes by 4, because the pixel is 32 bits

                    // copy the tile row
                    for (int j = 0; j < tilesPerRow; j++)
                    {
                        if (j != 0)
                        {
                            iColumnOrig += tileMargin.X;
                            iColumnNew += tileMargin.X;
                        }

                        // copy the leading pixel (tileBorder.X times)
                        iColumnNew = ClonePixel(dataOrig, dataNew, tileBorder, dataBytesOrig, iRowOrig, iColumnOrig, dataBytesNew, iRowNew, iColumnNew);

                        // copy the tile pixels
                        Buffer.BlockCopy(dataBytesOrig, iRowOrig * dataOrig.Stride + 4 * iColumnOrig,
                                         dataBytesNew, iRowNew * dataNew.Stride + 4 * iColumnNew,
                                         4 * tileSize.X);

                        iColumnOrig += tileSize.X;
                        iColumnNew += tileSize.X;


                        // copy the trailing pixel (tileBorder.X times)
                        iColumnNew = ClonePixel(dataOrig, dataNew, tileBorder, dataBytesOrig, iRowOrig, iColumnOrig - 1, dataBytesNew, iRowNew, iColumnNew);
                    }

                    // if first or last tile row was copied, duplicate it (tileBorder.Y times)
                    if (iTileY == 0 || iTileY == tileSize.Y - 1)
                    {
                        for (int iMargin = 0; iMargin < tileBorder.Y; iMargin++)
                        {
                            // copy the current row in dataNew to next row
                            Buffer.BlockCopy(dataBytesNew, iRowNew * dataNew.Stride, dataBytesNew,
                                (iRowNew + 1) * dataNew.Stride, dataNew.Stride);

                            iRowNew++;
                        }
                    }

                    iRowNew++;
                    iRowOrig++;
                }

            }

            System.Runtime.InteropServices.Marshal.Copy(dataBytesNew, 0, dataNewPtr, bytesNew);
        }

        private static int ClonePixel(BitmapData dataOrig, BitmapData dataNew, VRageMath.Vector2I tileBorder, byte[] dataBytesOrig,
            int iRowOrig, int iColumnOrig, byte[] dataBytesNew, int iRowNew, int iColumnNew)
        {
            for (int iMargin = 0; iMargin < tileBorder.X; iMargin++)
            {
                Buffer.BlockCopy(dataBytesOrig, iRowOrig * dataOrig.Stride + 4 * iColumnOrig,
                    dataBytesNew, iRowNew * dataNew.Stride + 4 * iColumnNew,
                    4);

                iColumnNew++;
            }
            return iColumnNew;
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
