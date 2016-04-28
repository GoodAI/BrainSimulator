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
                    int marginSizeIncrease = 4;

                    Bitmap bmpTextureWithBorders = new Bitmap(
                        tilesPerRow * (tilesetImage.TileSize.X + tilesetImage.TileMargin.X + marginSizeIncrease),
                        tilesPerColumn * (tilesetImage.TileSize.Y + tilesetImage.TileMargin.Y + marginSizeIncrease), 
                        System.Drawing.Imaging.PixelFormat.Format32bppArgb);

                    BitmapData dataOrig = bmp.LockBits(
                        new Rectangle(0, 0, bmp.Width, bmp.Height),
                        ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format32bppArgb);

                    BitmapData dataNew = bmpTextureWithBorders.LockBits(
                        new Rectangle(0, 0, bmpTextureWithBorders.Width, bmpTextureWithBorders.Height),
                        ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format32bppArgb);

                    // Get the address of the first line.
                    IntPtr dataOrigPtr = dataOrig.Scan0;
                    IntPtr dataNewPtr = dataNew.Scan0;

                    // Declare an array to hold the bytes of the bitmap.
                    int bytesOrig = Math.Abs(dataOrig.Stride) * bmp.Height;
                    int bytesNew = Math.Abs(dataNew.Stride) * bmpTextureWithBorders.Height;
                    byte[] dataBytesOrig = new byte[bytesOrig];
                    byte[] dataBytesNew = new byte[bytesNew];

                    // Copy the RGB values into the array.
                    System.Runtime.InteropServices.Marshal.Copy(dataOrigPtr, dataBytesOrig, 0, bytesOrig);

                    // increase the texture margins by 1...
                    for (int i = 0; i < dataOrig.Height; i++)
                    {
                        Buffer.BlockCopy(dataBytesOrig, i * dataOrig.Stride, dataBytesNew, i * dataNew.Stride, dataOrig.Stride);
                    }

                    int iRowOrig = 0;
                    int iRowNew = 0;

                    for (int i = 0; i < tilesPerColumn; i++)
                    {

                        // space between tiles
                        if(i != 0)
                        {
                            iRowOrig++;
                            iRowNew++;
                        }

                        // tiles
                        for (int iTileY = 0; iTileY < tilesetImage.TileSize.Y; iTileY++)
                        {
                            int iColumnOrig = 0;
                            int iColumnNew = 0;

                            // copy the tile row
                            for (int j = 0; j < tilesPerRow; j++)
                            {
                                if (j != 0)
                                {
                                    iColumnOrig++;
                                    iColumnNew++;
                                }

                                for (int iMargin = 0; iMargin < marginSizeIncrease / 2; iMargin++)
                                {

                                    Buffer.BlockCopy(dataBytesOrig, iRowOrig * dataOrig.Stride + 4 * iColumnOrig,
                                                     dataBytesNew, iRowNew * dataNew.Stride + 4 * iColumnNew,
                                                     4);

                                    //if (j != 0)
                                    {
                                        iColumnNew++;
                                    }
                                }

                                // 4 * because its 32bpp
                                Buffer.BlockCopy(dataBytesOrig, iRowOrig * dataOrig.Stride + 4 * iColumnOrig,
                                                 dataBytesNew, iRowNew * dataNew.Stride + 4 * iColumnNew,
                                                 4 * tilesetImage.TileSize.X);

                                iColumnOrig += tilesetImage.TileSize.X;
                                iColumnNew += tilesetImage.TileSize.X;

                                for (int iMargin = 0; iMargin < marginSizeIncrease / 2; iMargin++)
                                {
                                    Buffer.BlockCopy(dataBytesOrig, iRowOrig * dataOrig.Stride + 4 * (iColumnOrig - 1),  // -1 == last column
                                                     dataBytesNew, iRowNew * dataNew.Stride + 4 * iColumnNew,
                                                     4);

                                    iColumnNew++;
                                }
                            }

                            // if first or last tile row was copied, duplicate it
                            if(iTileY == 0 || iTileY == tilesetImage.TileSize.Y - 1)
                            {
                                for (int iMargin = 0; iMargin < marginSizeIncrease / 2; iMargin++)
                                {
                                    // copy the current row in dataNew to next row
                                    Buffer.BlockCopy(dataBytesNew, iRowNew * dataNew.Stride, dataBytesNew,
                                        (iRowNew + 1) * dataNew.Stride, dataNew.Stride);
                                    //if (!(iTileY == 0 && i == 0))
                                    {
                                        iRowNew++;
                                    }
                                }
                            }

                            iRowNew++;
                            iRowOrig++;
                        }
                        
                    }

                    System.Runtime.InteropServices.Marshal.Copy(dataBytesNew, 0, dataNewPtr, bytesNew);

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
