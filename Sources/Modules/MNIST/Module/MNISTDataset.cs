using System;
using System.IO;
using System.Linq;

namespace MNIST
{
    public class MNISTDatasetReader : IDatasetReader
    {
        public const int NumberOfClasses = 10;

        public const int ImageChannels = 1;
        public const int ImageRows = 28;
        public const int ImageColumns = 28;

        private const int ImageMagicNumber = 2051;
        private const int LabelMagicNumber = 2049;

        private BinaryReader m_brImages;
        private BinaryReader m_brLabels;

        private int m_nExamplesLeft;

        public int NumClasses { get { return NumberOfClasses; } }

        private static int readInt(BinaryReader br)
        {
            return BitConverter.ToInt32(br.ReadBytes(sizeof(int)).Reverse().ToArray(), 0);
        }

        public MNISTDatasetReader(string imageFilePath, string labelFilePath)
        {
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

            m_nExamplesLeft = readInt(m_brImages);

            if (m_nExamplesLeft != readInt(m_brLabels))
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

        public IExample ReadNext()
        {
            byte[] imageData = m_brImages.ReadBytes(ImageRows * ImageColumns);
            int label = m_brLabels.ReadByte();
            m_nExamplesLeft--;
            return new NormalizedExample(imageData, label);
        }

        public bool HasNext()
        {
            return m_nExamplesLeft > 0;
        }

        public void Dispose()
        {
            ((IDisposable)m_brImages).Dispose();
            ((IDisposable)m_brLabels).Dispose();
        }
    }

    public class MNISTDatasetReaderFactory : DatasetReaderFactory
    {
        private string m_baseDir;

        private const string TrainSetImageName = "train-images.idx3-ubyte";
        private const string TrainSetLabelName = "train-labels.idx1-ubyte";
        private const string TestSetImageName = "t10k-images.idx3-ubyte";
        private const string TestSetLabelName = "t10k-labels.idx1-ubyte";
        
        public MNISTDatasetReaderFactory(string baseDir, DatasetReaderFactoryType type)
            : base(type)
        {
            m_baseDir = baseDir;
        }

        protected override IDatasetReader CreateTrainReader()
        {
            return new MNISTDatasetReader(m_baseDir + TrainSetImageName, m_baseDir + TrainSetLabelName);
        }

        protected override IDatasetReader CreateTestReader()
        {
            return new MNISTDatasetReader(m_baseDir + TestSetImageName, m_baseDir + TestSetLabelName);
        }
    }
}
