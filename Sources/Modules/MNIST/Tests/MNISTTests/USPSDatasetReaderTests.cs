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
    public class USPSDatasetReaderTests
    {
        [Fact]
        public void Reads()
        {
            string path;
            float[][] imagesData;
            int[] labels;
            const int nExamples = 10;

            CreateUSPSFile(out path, out imagesData, out labels, nExamples);

            using (USPSDatasetReader reader = new USPSDatasetReader(path))
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

            File.Delete(path);
        }

        private static void CreateUSPSFile(out string filePath, out float[][] imagesData, out int[] labels, int nExamples)
        {
            Random rand = new Random();

            const int nFloatsPerImage = USPSDatasetReader.ImageRows * USPSDatasetReader.ImageColumns * USPSDatasetReader.ImageChannels;

            imagesData = new float[nExamples][];
            labels = new int[nExamples];

            filePath = Path.GetTempFileName();
            using (StreamWriter sw = new StreamWriter(filePath))
            {
                for (int i = 0; i < nExamples; ++i)
                {
                    labels[i] = rand.Next(byte.MaxValue);
                    sw.Write(labels[i]);

                    imagesData[i] = new float[nFloatsPerImage];
                    for (int j = 0; j < nFloatsPerImage; ++j)
                    {
                        imagesData[i][j] = (float)Math.Round(rand.NextDouble() * 2 - 1, 3);
                        sw.Write(' ');
                        sw.Write(imagesData[i][j]);
                    }
                    sw.Write('\n');
                }
            }
        }
    }
}
