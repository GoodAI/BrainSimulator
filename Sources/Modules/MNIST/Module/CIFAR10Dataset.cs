using System;
using System.IO;
using System.Linq;

namespace MNIST
{
    public class CIFAR10DatasetReader : DatasetReader
    {
        public new static string BaseDir => DatasetReader.BaseDir + @"CIFAR-10\";

        private static readonly string[] TrainFileNames = { "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin" };
        private static readonly string[] TestFileNames = { "test_batch.bin" };

        public static string[] DefaultTrainPaths => Array.ConvertAll(TrainFileNames, v => BaseDir + v);
        public static string[] DefaultTestPaths => Array.ConvertAll(TestFileNames, v => BaseDir + v);
        public static string[] DefaultNeededPaths => DefaultTrainPaths.Concat(DefaultTestPaths).ToArray();

        public const int NumberOfClasses = 10;

        public const int ImageChannels = 3;
        public const int ImageRows = 32;
        public const int ImageColumns = 32;

        private const int ExamplesPerDatasetPart = 10000;

        private string[] m_filePaths;
        private int m_fnIndex;

        private BinaryReader m_br;

        public override int NumClasses { get { return NumberOfClasses; } }

        public CIFAR10DatasetReader(string[] filePaths)
        {
            m_filePaths = filePaths;
            m_fnIndex = 0;

            if (!HasNextBatch())
            {
                throw new ArgumentException("Must provide at least one filepath for CIFAR-10 reader");
            }

            m_br = OpenNextDatasetpart();
        }

        private BinaryReader OpenNextDatasetpart()
        {
            FileStream ifs = new FileStream(m_filePaths[m_fnIndex++], FileMode.Open, FileAccess.Read);
            return new BinaryReader(ifs);
        }

        public override IExample ReadNext()
        {
            if (!CurrentBatchHasNext())
            {
                if (!HasNextBatch())
                {
                    throw new EndOfStreamException("Requested next CIFAR-10 example, but there are no more to be read (last file: \"" + m_filePaths[m_fnIndex] + "\"");
                }
                m_br.Dispose();
                m_br = OpenNextDatasetpart();
            }

            int label = m_br.ReadByte();
            byte[] imageData = m_br.ReadBytes(ImageRows * ImageColumns * ImageChannels);

            return new NormalizedExample(imageData, label);
        }

        private bool HasNextBatch()
        {
            return m_fnIndex < m_filePaths.Length;
        }

        private bool CurrentBatchHasNext()
        {
            Stream s = m_br.BaseStream;
            int bytesPerExample = ImageRows * ImageColumns * ImageChannels + 1; // + 1 byte for label
            return (s.Length - s.Position) >= bytesPerExample;
        }

        public override bool HasNext()
        {
            return HasNextBatch() || CurrentBatchHasNext();
        }

        public override void Dispose()
        {
            m_br.Dispose();
        }
    }

    public class CIFAR10DatasetReaderFactory : AbstractDatasetReaderFactory
    {
        private string[] m_filePaths;

        public CIFAR10DatasetReaderFactory(string[] filePaths)
        {
            m_filePaths = filePaths;
        }

        public override DatasetReader CreateReader()
        {
            return new CIFAR10DatasetReader(m_filePaths);
        }
    }
}
