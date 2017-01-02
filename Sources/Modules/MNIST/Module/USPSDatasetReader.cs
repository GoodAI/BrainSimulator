using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST
{
    public class USPSDatasetReader : IDatasetReader
    {
        public const int NumberOfClasses = 10;

        public const int ImageChannels = 1;
        public const int ImageRows = 16;
        public const int ImageColumns = 16;

        private StreamReader _sr;

        private int _nExamplesLeft;

        public int NumClasses { get { return NumberOfClasses; } }

        public USPSDatasetReader(string filePath)
        {
            _sr = new StreamReader(filePath);
        }

        public IExample ReadNext()
        {
            const int nPixels = ImageRows * ImageColumns;
            string line = _sr.ReadLine();
            string[] numbers = line.Split(new char[] {' '}, StringSplitOptions.RemoveEmptyEntries);

            if (numbers.Length != nPixels + 1)
            {
                //TODO: include path of the file?
                throw new InvalidDataException("Invalid USPS file: invalid number of elements on line");
            }
            var ci = new CultureInfo("en-US");

            int target = (int) float.Parse(numbers[0],ci);
            float[] input = new float[nPixels];
            for (int i = 0; i < nPixels; ++i)
            {
                input[i] = float.Parse(numbers[i + 1],ci);
            }

            return new Example(input, target);
        }

        public bool HasNext()
        {
            return !_sr.EndOfStream;
        }

        public void Dispose()
        {
            ((IDisposable)_sr).Dispose();
        }
    }

    public class USPSDatasetReaderFactory : DatasetReaderFactory
    {
        private string _baseDir;

        private const string TrainSetName = "zip.train";
        private const string TestSetName = "zip.test";
        
        public USPSDatasetReaderFactory(string baseDir, DatasetReaderFactoryType type)
            : base(type)
        {
            _baseDir = baseDir;
        }

        protected override IDatasetReader CreateTrainReader()
        {
            return new USPSDatasetReader(_baseDir + TrainSetName);
        }

        protected override IDatasetReader CreateTestReader()
        {
            return new USPSDatasetReader(_baseDir + TestSetName);
        }
    }
}
