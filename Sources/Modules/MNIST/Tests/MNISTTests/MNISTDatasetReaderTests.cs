using MNIST;
using System;
using System.IO;
using System.Linq;
using Xunit;

namespace MNISTTests
{
    public class MNISTDatasetReaderTests
    {
        [Fact]
        public void Reads()
        {
            string imagePath, labelPath;
            byte[][] imagesData;
            int[] labels;
            const int nExamples = 10;

            CreateMNISTfiles(out imagePath, out labelPath, out imagesData, out labels, nExamples);

            using (MNISTDatasetReader reader = new MNISTDatasetReader(imagePath, labelPath))
            {
                for (int i = 0; i < nExamples; ++i)
                {
                    Assert.True(reader.HasNext(), "Reader should not be at the end");

                    IExample refExample = new NormalizedExample(imagesData[i], labels[i]);
                    IExample readExample = reader.ReadNext();

                    Assert.True(Enumerable.SequenceEqual(refExample.Input, readExample.Input), "Written and read image data should be the same");
                    Assert.True(refExample.Target == readExample.Target, "Written and read labels should be the same");
                }

                Assert.False(reader.HasNext(), "Reader should be at the end");
            }

            File.Delete(imagePath);
            File.Delete(labelPath);
        }

        private static void WriteInt(BinaryWriter bw, int val)
        {
            bw.Write(BitConverter.GetBytes(val).Reverse().ToArray());
        }

        private static BinaryWriter OpenForWrite(string path)
        {
            FileStream ifs = new FileStream(path, FileMode.Open, FileAccess.Write);
            return new BinaryWriter(ifs);
        }

        private static void WriteImageHeader(BinaryWriter bw)
        {
            WriteInt(bw, MNISTDatasetReader.ImageMagicNumber);
            WriteInt(bw, 2);
            WriteInt(bw, MNISTDatasetReader.ImageRows);
            WriteInt(bw, MNISTDatasetReader.ImageColumns);
        }

        private static void WriteLabelHeader(BinaryWriter bw)
        {
            WriteInt(bw, MNISTDatasetReader.LabelMagicNumber);
            WriteInt(bw, 2);
        }

        private static void CreateMNISTfiles(out string imagePath, out string labelPath, out byte[][] imagesData, out int[] labels, int nExamples)
        {
            Random rand = new Random();

            const int nBytesPerImage = MNISTDatasetReader.ImageRows * MNISTDatasetReader.ImageColumns * MNISTDatasetReader.ImageChannels;

            imagesData = new byte[nExamples][];
            labels= new int[nExamples];

            imagePath = Path.GetTempFileName();
            using (BinaryWriter bw = OpenForWrite(imagePath))
            {
                WriteImageHeader(bw);
                for (int i = 0; i < nExamples; ++i)
                {
                    imagesData[i] = new byte[nBytesPerImage];
                    rand.NextBytes(imagesData[i]);
                    bw.Write(imagesData[i]);
                }
            }

            labelPath = Path.GetTempFileName();
            using (BinaryWriter bw = OpenForWrite(labelPath))
            {
                WriteLabelHeader(bw);
                for (int i = 0; i < nExamples; ++i)
                {
                    labels[i] = rand.Next(byte.MaxValue);
                    bw.Write((byte)labels[i]);
                }
            }
        }
    }
}
