using System;
using System.IO;
using System.Linq;

namespace MNIST
{
    public class MNISTDatasetReader : DatasetReader
    {
        public new static string BaseDir => DatasetReader.BaseDir + @"MNIST\";

        public static string DefaultTrainImagePath => BaseDir + "train-images.idx3-ubyte";
        public static string DefaultTestImagePath => BaseDir + "t10k-images.idx3-ubyte";
        public static string DefaultTrainLabelPath => BaseDir + "train-labels.idx1-ubyte";
        public static string DefaultTestLabelPath => BaseDir + "t10k-labels.idx1-ubyte";
        public static string[] DefaultNeededPaths => new string[] { DefaultTrainImagePath, DefaultTrainLabelPath, DefaultTestImagePath, DefaultTestLabelPath };

        public const int NumberOfClasses = 10;

        public const int ImageChannels = 1;
        public const int ImageRows = 28;
        public const int ImageColumns = 28;

        private const int ImageMagicNumber = 2051;
        private const int LabelMagicNumber = 2049;

        private BinaryReader m_brImages;
        private BinaryReader m_brLabels;

        private string m_imageFilePath;
        private string m_labelFilePath;

        public override int NumClasses { get { return NumberOfClasses; } }

        private static int readInt(BinaryReader br)
        {
            return BitConverter.ToInt32(br.ReadBytes(sizeof(int)).Reverse().ToArray(), 0);
        }

        public MNISTDatasetReader(string imageFilePath, string labelFilePath)
        {
            m_imageFilePath = imageFilePath;
            m_labelFilePath = labelFilePath;

            FileStream ifsImages = new FileStream(imageFilePath, FileMode.Open, FileAccess.Read);
            FileStream ifsLabels = new FileStream(labelFilePath, FileMode.Open, FileAccess.Read);

            m_brImages = new BinaryReader(ifsImages);
            m_brLabels = new BinaryReader(ifsLabels);

            if (readInt(m_brImages) != ImageMagicNumber)
            {
                throw new InvalidDataException("MNIST file \"" + imageFilePath + "\" magic number mismatch");
            }

            if (readInt(m_brLabels) != LabelMagicNumber)
            {
                throw new InvalidDataException("MNIST file \"" + labelFilePath + "\" magic number mismatch");
            }

            int nExamples = readInt(m_brImages);
            if (readInt(m_brLabels) != nExamples)
            {
                throw new InvalidDataException("Number of examples does not match number of labels in MNIST dataset");
            }

            if (readInt(m_brImages) != ImageRows)
            {
                throw new InvalidDataException("MNIST file \"" + imageFilePath + "\" rows number mismatch");
            }

            if (readInt(m_brImages) != ImageColumns)
            {
                throw new InvalidDataException("MNIST file \"" + imageFilePath + "\" columns number mismatch");
            }
        }

        public override IExample ReadNext()
        {
            if (!HasNextImage())
            {
                throw new EndOfStreamException("Reached end of the MNIST file \"" + m_imageFilePath + "\" while trying to read next example");
            }

            if (!HasNextLabel())
            {
                throw new EndOfStreamException("Reached end of the MNIST file \"" + m_labelFilePath + "\" while trying to read next example");
            }

            byte[] imageData = m_brImages.ReadBytes(ImageRows * ImageColumns);
            int label = m_brLabels.ReadByte();
            return new NormalizedExample(imageData, label);
        }

        private static bool HasBytesToRead(BinaryReader br, int nBytes)
        {
            Stream s = br.BaseStream;
            return (s.Length - s.Position) >= nBytes;
        }

        private bool HasNextImage()
        {
            return HasBytesToRead(m_brImages, ImageRows * ImageColumns);
        }

        private bool HasNextLabel()
        {
            return HasBytesToRead(m_brLabels, 1);
        }

        public override bool HasNext()
        {
            return HasNextImage() && HasNextLabel();
        }

        public override void Dispose()
        {
            m_brImages.Dispose();
            m_brLabels.Dispose();
        }
    }

    public class MNISTDatasetReaderFactory : AbstractDatasetReaderFactory
    {
        private string m_imageFilePath;
        private string m_labelFilePath;

        public MNISTDatasetReaderFactory(string imageFilePath, string labelFilePath)
        {
            m_imageFilePath = imageFilePath;
            m_labelFilePath = labelFilePath;
        }

        public override DatasetReader CreateReader()
        {
            return new MNISTDatasetReader(m_imageFilePath, m_labelFilePath);
        }
    }
}
