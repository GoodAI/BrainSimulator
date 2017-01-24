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

        private StreamReader m_sr;
        private string m_filePath;

        public int NumClasses { get { return NumberOfClasses; } }

        public USPSDatasetReader(string filePath)
        {
            m_sr = new StreamReader(filePath);
            m_filePath = filePath;
        }

        public IExample ReadNext()
        {
            const int nPixels = ImageRows * ImageColumns;
            string line = m_sr.ReadLine();

            if (line == null)
            {
                throw new EndOfStreamException("Reached end of the USPS file \"" + m_filePath + "\" while trying to read next example");
            }

            string[] numbers = line.Split(new char[] {' '}, StringSplitOptions.RemoveEmptyEntries);

            if (numbers.Length != nPixels + 1)
            {
                throw new InvalidDataException("Invalid USPS file \"" + m_filePath + "\": invalid number of elements per line");
            }

            CultureInfo ci = new CultureInfo("en-US");
            int label = (int) float.Parse(numbers[0], ci);

            float[] imageData = new float[nPixels];
            for (int i = 0; i < nPixels; ++i)
            {
                imageData[i] = float.Parse(numbers[i + 1], ci);
            }

            return new NormalizedExample(imageData, label);
        }

        public bool HasNext()
        {
            return !m_sr.EndOfStream;
        }

        public void Dispose()
        {
            m_sr.Dispose();
        }
    }

    public class USPSDatasetTrainReaderFactory : AbstractDatasetReaderFactory
    {
        private const string FileName = "zip.train";
        private string m_baseDir;

        public USPSDatasetTrainReaderFactory(string baseDir)
        {
            m_baseDir = baseDir;
        }

        public override IDatasetReader CreateReader()
        {
            return new USPSDatasetReader(m_baseDir + FileName);
        }
    }

    public class USPSDatasetTestReaderFactory : AbstractDatasetReaderFactory
    {
        private const string FileName = "zip.test";
        private string m_baseDir;

        public USPSDatasetTestReaderFactory(string baseDir)
        {
            m_baseDir = baseDir;
        }

        public override IDatasetReader CreateReader()
        {
            return new USPSDatasetReader(m_baseDir + FileName);
        }
    }
}
