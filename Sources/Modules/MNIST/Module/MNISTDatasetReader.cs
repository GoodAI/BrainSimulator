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

        private BinaryReader _brImages;
        private BinaryReader _brLabels;

        private int _nExamplesLeft;

        public int NumClasses { get { return NumberOfClasses; } }

        private static int readInt(BinaryReader br)
        {
            return BitConverter.ToInt32(br.ReadBytes(sizeof(int)).Reverse().ToArray(), 0);
        }

        public MNISTDatasetReader(string imageFilePath, string labelFilePath)
        {
            FileStream ifsImages = new FileStream(imageFilePath, FileMode.Open, FileAccess.Read);
            FileStream ifsLabels = new FileStream(labelFilePath, FileMode.Open, FileAccess.Read);

            _brImages = new BinaryReader(ifsImages);
            _brLabels = new BinaryReader(ifsLabels);

            if (readInt(_brImages) != ImageMagicNumber)
            {
                throw new InvalidDataException("MNIST file \"" + imageFilePath + "\" magic number mismatch");
            }
            
            if (readInt(_brLabels) != LabelMagicNumber)
            {
                throw new InvalidDataException("MNIST file \"" + labelFilePath + "\" magic number mismatch");
            }

            _nExamplesLeft = readInt(_brImages);

            if (_nExamplesLeft != readInt(_brLabels))
            {
                throw new InvalidDataException("Number of examples does not match number of labels in MNIST dataset");
            }

            if (readInt(_brImages) != ImageRows)
            {
                throw new InvalidDataException("MNIST file \"" + imageFilePath + "\" rows number mismatch");
            }

            if (readInt(_brImages) != ImageColumns)
            {
                throw new InvalidDataException("MNIST file \"" + imageFilePath + "\" columns number mismatch");
            }
        }

        public IExample ReadNext()
        {
            byte[] imageData = _brImages.ReadBytes(ImageRows * ImageColumns);
            int label = _brLabels.ReadByte();
            _nExamplesLeft--;
            return new Example(imageData, label);
        }

        public bool HasNext()
        {
            return _nExamplesLeft > 0;
        }

        public void Dispose()
        {
            ((IDisposable)_brImages).Dispose();
            ((IDisposable)_brLabels).Dispose();
        }
    }

    public class MNISTDatasetReaderFactory : DatasetReaderFactory
    {
        private string _baseDir;

        private const string TrainSetImageName = "train-images.idx3-ubyte";
        private const string TrainSetLabelName = "train-labels.idx1-ubyte";
        private const string TestSetImageName = "t10k-images.idx3-ubyte";
        private const string TestSetLabelName = "t10k-labels.idx1-ubyte";
        
        public MNISTDatasetReaderFactory(string baseDir, DatasetReaderFactoryType type)
            : base(type)
        {
            _baseDir = baseDir;
        }

        protected override IDatasetReader CreateTrainReader()
        {
            return new MNISTDatasetReader(_baseDir + TrainSetImageName, _baseDir + TrainSetLabelName);
        }

        protected override IDatasetReader CreateTestReader()
        {
            return new MNISTDatasetReader(_baseDir + TestSetImageName, _baseDir + TestSetLabelName);
        }
    }
}
