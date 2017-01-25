using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST
{
    public class USPSDatasetReader : DatasetReader
    {
        public new static string BaseDir => DatasetReader.BaseDir + @"USPS\";

        public static string DefaultTrainPath => BaseDir + "zip.train";
        public static string DefaultTestPath => BaseDir + "zip.test";
        public static string[] DefaultNeededPaths => new string[] { DefaultTrainPath, DefaultTestPath };

        public const int NumberOfClasses = 10;

        public const int ImageChannels = 1;
        public const int ImageRows = 16;
        public const int ImageColumns = 16;

        private StreamReader m_sr;
        private string m_filePath;

        public override int NumClasses { get { return NumberOfClasses; } }

        public USPSDatasetReader(string filePath)
        {
            m_sr = new StreamReader(filePath);
            m_filePath = filePath;
        }

        public override IExample ReadNext()
        {
            if (!HasNext())
            {
                throw new EndOfStreamException("Reached end of the USPS file \"" + m_filePath + "\" while trying to read next example");
            }

            const int nPixels = ImageRows * ImageColumns;
            string line = m_sr.ReadLine();

            string[] numbers = line.Split(new char[] {' '}, StringSplitOptions.RemoveEmptyEntries);

            if (numbers.Length != nPixels + 1)
            {
                throw new InvalidDataException("Invalid USPS file \"" + m_filePath + "\": invalid number of elements per line");
            }

            int label = (int) float.Parse(numbers[0], CultureInfo.InvariantCulture);

            float[] imageData = new float[nPixels];
            for (int i = 0; i < nPixels; ++i)
            {
                imageData[i] = float.Parse(numbers[i + 1], CultureInfo.InvariantCulture);
            }

            return new NormalizedExample(imageData, label);
        }

        public override bool HasNext()
        {
            return !m_sr.EndOfStream;
        }

        public override void Dispose()
        {
            m_sr.Dispose();
        }
    }

    public class USPSDatasetReaderFactory : AbstractDatasetReaderFactory
    {
        private string m_filePath;

        public USPSDatasetReaderFactory(string filePath)
        {
            m_filePath = filePath;
        }

        public override DatasetReader CreateReader()
        {
            return new USPSDatasetReader(m_filePath);
        }
    }
}
