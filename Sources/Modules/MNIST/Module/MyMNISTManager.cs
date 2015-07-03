using System;
using System.Collections;
using System.IO;
using System.Linq;

/*
 * Import of MNIST done by: http://jamesmccaffrey.wordpress.com/2013/11/23/reading-the-mnist-data-set-with-c/
 * and http://msdn.microsoft.com/en-us/magazine/dn745868.aspx
 * MNIST data available at http://yann.lecun.com/exdb/mnist/
 */

namespace MNIST
{
    public enum MNISTLastImageMethod
    {
        ResetToStart,
        SendNothing
    }

    public class MyMNISTManager
    {
        private int m_imagesServed;
        public int m_imagesDemand;
        private string m_baseFolder;
        private ArrayList m_images;
        private MyMNISTImage m_blankImage;
        private IEnumerator m_imageEnumerator;
        private MNISTLastImageMethod m_afterLastImage;

        public int m_sequenceIterator;
        public bool m_definedOrder;

        public bool RandomEnumerate = false;
        private Random rand = new Random();

        /// <summary>
        /// Converts between little-endian and big-endian
        /// </summary>
        /// <param name="value">Value to convert</param>
        /// <returns></returns>
        public static int ReverseBytes(int value)
        {
            byte[] intAsBytes = BitConverter.GetBytes(value);
            Array.Reverse(intAsBytes);
            return BitConverter.ToInt32(intAsBytes, 0);
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="baseFolder">Base folder with MNIST dataset files</param>
        /// <param name="imagesCnt">How many images to load</param>
        /// <param name="exact">If TRUE, you will get exactly imagesCnt images. If FALSE, you will get AT MOST imagesCnt images. It's here for performance reason and BC</param>
        /// <param name="afterLastImage">What to do, after the last values has been sent</param>
        public MyMNISTManager(string baseFolder, int imagesCnt = int.MaxValue, bool exact = false, MNISTLastImageMethod afterLastImage = MNISTLastImageMethod.ResetToStart)
        {
            m_baseFolder = baseFolder;
            m_images = new ArrayList();
            m_afterLastImage = afterLastImage;
            m_imagesDemand = imagesCnt;
            m_imagesServed = 0;
            m_sequenceIterator = 0;
            m_definedOrder = false;

            FileStream ifsTrainLabels = new FileStream(m_baseFolder + "train-labels.idx1-ubyte", FileMode.Open, FileAccess.Read);
            FileStream ifsTrainImages = new FileStream(m_baseFolder + "train-images.idx3-ubyte", FileMode.Open, FileAccess.Read);

            BinaryReader brTrainLabels = new BinaryReader(ifsTrainLabels);
            BinaryReader brTrainImages = new BinaryReader(ifsTrainImages);

            //Magic number
            brTrainLabels.ReadInt32();
            brTrainImages.ReadInt32();

            int numImagesTrainLables = brTrainLabels.ReadInt32();
            numImagesTrainLables = ReverseBytes(numImagesTrainLables);
            int numImagesTrainImages = brTrainImages.ReadInt32();
            numImagesTrainImages = ReverseBytes(numImagesTrainImages);

            int numRowsTrainImages = brTrainImages.ReadInt32();
            numRowsTrainImages = ReverseBytes(numRowsTrainImages);
            int numColsTrainImages = brTrainImages.ReadInt32();
            numColsTrainImages = ReverseBytes(numColsTrainImages);

            int maxImages;
            if (exact)
            {
                // numImagesTrainImages = 60000
                maxImages = numImagesTrainImages;
            }
            else
            {
                // value of 2000 is a compromise between long loading-time for maximum of 60k images and minimum of user set "imagesCnt"
                // this also brings more flexibility for changing the m_imagesDemand during simulation (up to max(2000,imagesCnt) instead of imagesCnt)
                maxImages = Math.Max(2000, Math.Min(numImagesTrainImages, imagesCnt));
            }

            byte[][] data = new byte[numColsTrainImages][];
            for (int i = 0; i < data.Length; ++i)
                data[i] = new byte[numRowsTrainImages];

            for (int i = 0; i < maxImages; ++i)
            {
                for (int j = 0; j < numColsTrainImages; ++j)
                {
                    for (int k = 0; k < numRowsTrainImages; ++k)
                    {
                        byte b = brTrainImages.ReadByte();
                        data[j][k] = b;
                    }
                }

                byte label = brTrainLabels.ReadByte();
                MyMNISTImage mImage = new MyMNISTImage(data, label, numColsTrainImages, numRowsTrainImages);
                m_images.Add(mImage);
            }

            ifsTrainImages.Close();
            brTrainImages.Close();
            ifsTrainLabels.Close();
            brTrainLabels.Close();

            m_imageEnumerator = m_images.GetEnumerator();

            for (int i = 0; i < numColsTrainImages; ++i)
                for (int j = 0; j < numRowsTrainImages; ++j)
                    data[i][j] = 0;
            m_blankImage = new MyMNISTImage(data, 0);
        }

        /// <summary>
        /// Gets the next values
        /// </summary>
        /// <param name="validNumbers">Array of integers, you want the selection restrict to.</param>
        /// <returns>Array of arrays of floats, in which the values is encoded.</returns>
        public MyMNISTImage GetNextImage(int[] validNumbers)
        {
            if (RandomEnumerate)
            {
                MyMNISTImage im = (MyMNISTImage)m_images[rand.Next(Math.Min(m_imagesDemand, m_images.Count))];

                if (m_definedOrder && im.Label != validNumbers[m_sequenceIterator] || !validNumbers.Contains(im.Label))
                {
                    return this.GetNextImage(validNumbers);
                }

                m_sequenceIterator = (m_sequenceIterator + 1) % validNumbers.Length;
                return im;
            }
            else if (m_imageEnumerator.MoveNext() && m_imagesServed < m_imagesDemand)
            {
                MyMNISTImage im = (MyMNISTImage)m_imageEnumerator.Current;
                m_imagesServed++;

                if (m_definedOrder)
                {
                    if (im.Label != validNumbers[m_sequenceIterator])
                    {
                        return this.GetNextImage(validNumbers);
                    }
                    m_sequenceIterator = (m_sequenceIterator + 1) % validNumbers.Length;
                }
                else
                {
                    if (!validNumbers.Contains(im.Label))
                    {
                        return this.GetNextImage(validNumbers);
                    }
                }
                return im;
            }
            else
            {
                switch (m_afterLastImage)
                {
                    case MNISTLastImageMethod.ResetToStart:
                        {
                            m_imageEnumerator.Reset();
                            m_imagesServed = 0; // Hack
                            return GetNextImage(validNumbers);
                        }
                    case MNISTLastImageMethod.SendNothing:
                        {
                            return m_blankImage;
                        }
                    default:
                        {
                            return GetNextImage(validNumbers);
                        }
                }
            }
        }
    }

    public class MyMNISTImage
    {
        private int m_width;
        private int m_height;
        private byte m_label;
        private float[][] m_data;
        private float[] m_data1d;
        private float m_min;
        private float m_max;

        public int Label
        {
            get { return m_label; }
        }

        public float[][] Data
        {
            get { return m_data; }
        }

        public float[] Data1D
        {
            get { return m_data1d; }
        }

        private void Normalize()
        {
            m_max = 0;
            m_min = float.MaxValue;

            for (int i = 0; i < m_width; ++i)
            {
                for (int j = 0; j < m_height; ++j)
                {
                    if (m_max < m_data[i][j]) { m_max = m_data[i][j]; }
                    if (m_min > m_data[i][j]) { m_min = m_data[i][j]; }
                }
            }

            if (m_min == 0 && m_max == 0) return;
            for (int i = 0; i < m_width; ++i)
            {
                for (int j = 0; j < m_height; ++j)
                {
                    m_data[i][j] = (m_data[i][j] - m_min) / (m_max - m_min);
                }
            }
        }

        public MyMNISTImage(byte[][] data, byte label, int width = 28, int height = 28)
        {
            m_width = width;
            m_height = height;
            m_label = label;
            m_data = new float[m_width][];

            for (int i = 0; i < m_data.Length; ++i)
            {
                m_data[i] = new float[m_height];
                for (int j = 0; j < m_height; ++j)
                    m_data[i][j] = (float)data[i][j];
            }

            Normalize();

            m_data1d = new float[m_height * m_width];
            int idx = 0;
            for (int j = 0; j < m_width; ++j)
                for (int k = 0; k < m_height; ++k)
                    m_data1d[idx++] = m_data[j][k]; //read from m_data, so it's also normalized
        }

        public void ToBinary()
        {
            m_data1d = new float[m_height * m_width];
            int idx = 0;
            for (int i = 0; i < m_width; ++i)
                for (int j = 0; j < m_height; ++j)
                {
                    if (m_data[i][j] < 0.5) { m_data1d[idx++] = 0; }
                    else { m_data1d[idx++] = 1; }
                }
        }
    }
}