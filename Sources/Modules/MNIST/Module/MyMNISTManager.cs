using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GoodAI.Core.Utils;

using System.Diagnostics;

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

    public enum MNISTSetType
    {
        Training,
        Test
    }

    public class MyMNISTManager
    {
        private int m_imagesServed;
        public int m_trainingExamplesPerDigitCnt;
        public int m_testExamplesPerDigitCnt;
        private string m_baseFolder;
        private ArrayList m_trainingImages;
        private ArrayList m_testImages;
        private MyMNISTImage m_blankImage;
        private IEnumerator m_trainingImagesEnumerator;
        private IEnumerator m_testImagesEnumerator;
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
        public MyMNISTManager(string baseFolder, int trainingExamplesPerDigitCnt = int.MaxValue, int testExamplesPerDigitCnt = int.MaxValue,
            bool exact = false, MNISTLastImageMethod afterLastImage = MNISTLastImageMethod.ResetToStart)
        {
            m_baseFolder = baseFolder;
            m_trainingImages = new ArrayList();
            m_testImages = new ArrayList();
            m_afterLastImage = afterLastImage;
            m_trainingExamplesPerDigitCnt = trainingExamplesPerDigitCnt;
            m_testExamplesPerDigitCnt = testExamplesPerDigitCnt;
            m_imagesServed = 0;
            m_sequenceIterator = 0;
            m_definedOrder = false;

            // START TIME MEASUREMENT ----------------------------------------------------------
            //Stopwatch sw = new Stopwatch();
            //sw.Start();
            ReadMnistSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte", m_trainingExamplesPerDigitCnt, m_trainingImages);
            ReadMnistSet("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", m_testExamplesPerDigitCnt, m_testImages);

            byte[,] data = new byte[28, 28];
            m_blankImage = new MyMNISTImage(data, 0);

            m_trainingImagesEnumerator = m_trainingImages.GetEnumerator();
            m_testImagesEnumerator = m_testImages.GetEnumerator();

            //sw.Stop();
            //Console.WriteLine("Elapsed={0}", sw.Elapsed);
        }

        private void ReadMnistSet(String imagesInputFile, String labelsInputFile, int numDigitClasses, ArrayList images)
        {
            FileStream ifsLabels = new FileStream(m_baseFolder + labelsInputFile, FileMode.Open, FileAccess.Read);
            FileStream ifsImages = new FileStream(m_baseFolder + imagesInputFile, FileMode.Open, FileAccess.Read);

            BinaryReader brLabels = new BinaryReader(ifsLabels);
            BinaryReader brImages = new BinaryReader(ifsImages);

            //Magic number
            brLabels.ReadInt32();
            brImages.ReadInt32();

            int numImagesLables = brLabels.ReadInt32();
            numImagesLables = ReverseBytes(numImagesLables);
            int numImages = brImages.ReadInt32();
            numImages = ReverseBytes(numImages);

            int numRowsImages = brImages.ReadInt32();
            numRowsImages = ReverseBytes(numRowsImages);
            int numColsImages = brImages.ReadInt32();
            numColsImages = ReverseBytes(numColsImages);

            int numOfLoadedDigitClasses = 0;
            int[] digitCounts = new int[10];    // helper array used for loading a given number of each digit

            if (numImagesLables == numImages)
            {
                for (int i = 0; i < numImages; ++i)
                {
                    MyMNISTImage mImage = new MyMNISTImage(brImages, brLabels, numColsImages, numRowsImages);
                    if (digitCounts[mImage.Label] < numDigitClasses)
                    {
                        images.Add(mImage);
                        digitCounts[mImage.Label]++;
                        if (digitCounts[mImage.Label] == numDigitClasses)
                        {
                            numOfLoadedDigitClasses++;
                        }
                    }
                    if (numOfLoadedDigitClasses == 10)
                    {
                        break;
                    }
                }
            }

            ifsImages.Close();
            brImages.Close();
            ifsLabels.Close();
            brLabels.Close();
        }

        /// <summary>
        /// Gets the next values
        /// </summary>
        /// <param name="validNumbers">Array of integers, you want the selection restrict to.</param>
        /// <returns>Array of arrays of floats, in which the values is encoded.</returns>
        public MyMNISTImage GetNextImage(int[] validNumbers, MNISTSetType setType)
        {
            ArrayList images = null;
            IEnumerator enumerator = null;
            int examplesPerDigitCnt = 0;
            if (setType == MNISTSetType.Training)
            {
                images = m_trainingImages;
                enumerator = m_trainingImagesEnumerator;
                examplesPerDigitCnt = m_trainingExamplesPerDigitCnt;
            }
            else if (setType == MNISTSetType.Test)
            {
                images = m_testImages;
                enumerator = m_testImagesEnumerator;
                examplesPerDigitCnt = m_testExamplesPerDigitCnt;
            }
            
            if (RandomEnumerate)
            {
                MyMNISTImage im = (MyMNISTImage)images[rand.Next(Math.Min(examplesPerDigitCnt * 10, images.Count))];

                // choose random image until it is in the validNumbers set;
                // if "sequence ordered" is ON it also has to be the expected next number in given sequence
                if (m_definedOrder && im.Label != validNumbers[m_sequenceIterator] || !validNumbers.Contains(im.Label))
                {
                    return this.GetNextImage(validNumbers, setType);
                }

                m_sequenceIterator = (m_sequenceIterator + 1) % validNumbers.Length;
                return im;
            }
            else if (enumerator.MoveNext() && m_imagesServed < images.Count)
            {
                MyMNISTImage im = (MyMNISTImage)enumerator.Current;
                m_imagesServed++;

                if (m_definedOrder)
                {
                    if (im.Label != validNumbers[m_sequenceIterator])
                    {
                        return GetNextImage(validNumbers, setType);
                    }
                    m_sequenceIterator = (m_sequenceIterator + 1) % validNumbers.Length;
                }
                else
                {
                    if (!validNumbers.Contains(im.Label))
                    {
                        return GetNextImage(validNumbers, setType);
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
                            enumerator.Reset();
                            m_imagesServed = 0; // Hack
                            return GetNextImage(validNumbers, setType);
                        }
                    case MNISTLastImageMethod.SendNothing:
                        {
                            return m_blankImage;
                        }
                    default:
                        {
                            return GetNextImage(validNumbers, setType);
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
        private float[,] m_data;
        private float[] m_data1d;
        private float m_min;
        private float m_max;

        public int Label
        {
            get { return m_label; }
        }

        public float[,] Data
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
                    if (m_max < m_data[i, j]) { m_max = m_data[i, j]; }
                    if (m_min > m_data[i, j]) { m_min = m_data[i, j]; }
                }
            }

            if (m_min == 0 && m_max == 0) return;
            for (int i = 0; i < m_width; ++i)
            {
                for (int j = 0; j < m_height; ++j)
                {
                    m_data[i, j] = (m_data[i, j] - m_min) / (m_max - m_min);
                }
            }
        }

        public MyMNISTImage(byte[,] data, byte label, int width = 28, int height = 28)
        {
            m_width = width;
            m_height = height;
            m_label = label;
            m_data = new float[m_width, m_height];

            for (int i = 0; i < m_width; ++i)
            {
                for (int j = 0; j < m_height; ++j)
                    m_data[i, j] = (float)data[i, j];
            }

            Normalize();

            m_data1d = new float[m_height * m_width];
            int idx = 0;
            for (int j = 0; j < m_width; ++j)
                for (int k = 0; k < m_height; ++k)
                    m_data1d[idx++] = m_data[j, k]; //read from m_data, so it's also normalized
        }

        /// <summary>
        /// Alternative optimized constructor; Directly reads from referenced BinaryReaders into inner data store m_data.
        /// The normalization into <0, 1> is also performed right away with respect to min = 0, max = 255 of the read byte value.
        /// </summary>
        public MyMNISTImage(BinaryReader brImages, BinaryReader brLabels, int width = 28, int height = 28)
        {
            m_max = 255.0f;
            m_min = 0.0f;
            m_width = width;
            m_height = height;

            m_data = new float[m_width, m_height];
            m_data1d = new float[m_width * m_height];
            float normalizedValue = 0.0f;
            int idx = 0;

            for (int i = 0; i < m_width; ++i)
            {
                for (int j = 0; j < m_height; ++j)
                {
                    normalizedValue = ((float)brImages.ReadByte() - m_min) / (m_max - m_min);
                    m_data[i, j] = normalizedValue;
                    m_data1d[idx++] = normalizedValue;
                }
            }
            m_label = brLabels.ReadByte();
        }

        public void ToBinary()
        {
            m_data1d = new float[m_height * m_width];
            int idx = 0;
            for (int i = 0; i < m_width; ++i)
                for (int j = 0; j < m_height; ++j)
                {
                    if (m_data[i, j] < 0.5) { m_data1d[idx++] = 0; }
                    else { m_data1d[idx++] = 1; }
                }
        }
    }
}