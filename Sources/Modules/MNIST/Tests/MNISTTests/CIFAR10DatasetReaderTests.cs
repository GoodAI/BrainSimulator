using MNIST;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace MNISTTests
{
    public class CIFAR10DatasetReaderTests
    {
        [Fact]
        public void Reads()
        {
            string[] paths;
            byte[][] imagesData;
            int[] labels;
            const int nFiles = 3;
            const int nExamplesPerFile = 4;
            const int nExamples = nFiles * nExamplesPerFile;

            CreateCIFAR10Files(out paths, out imagesData, out labels, nFiles, nExamplesPerFile);

            using (CIFAR10DatasetReader reader = new CIFAR10DatasetReader(paths))
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

            foreach (string path in paths)
            {
                File.Delete(path);
            }
        }

        private static BinaryWriter OpenForWrite(string path)
        {
            FileStream ifs = new FileStream(path, FileMode.Open, FileAccess.Write);
            return new BinaryWriter(ifs);
        }

        private static void CreateCIFAR10Files(out string[] filePaths, out byte[][] imagesData, out int[] labels, int nFiles, int nExamplesPerFile)
        {
            Random rand = new Random();

            const int nBytesPerImage = CIFAR10DatasetReader.ImageRows * CIFAR10DatasetReader.ImageColumns * CIFAR10DatasetReader.ImageChannels;
            int nExamples = nFiles * nExamplesPerFile;

            filePaths = new string[nFiles];
            imagesData = new byte[nExamples][];
            labels = new int[nExamples];

            for (int i = 0; i < nFiles; ++i)
            {
                filePaths[i] = Path.GetTempFileName();
                using (BinaryWriter bw = new BinaryWriter(File.OpenWrite(filePaths[i])))
                {
                    for (int j = 0; j < nExamplesPerFile; ++j)
                    {
                        int exampleIndex = i * nExamplesPerFile + j;

                        labels[exampleIndex] = rand.Next(byte.MaxValue);
                        bw.Write((byte)labels[exampleIndex]);

                        imagesData[exampleIndex] = new byte[nBytesPerImage];
                        rand.NextBytes(imagesData[exampleIndex]);
                        bw.Write(imagesData[exampleIndex]);
                    }
                }
            }
        }
    }
}
