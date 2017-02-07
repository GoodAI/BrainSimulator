using MNIST;
using System;
using System.Linq;
using Xunit;

namespace MNISTTests
{
    public class NormalizedExampleTests
    {
        private const int DataLen = 256;
        private Random m_rand = new Random();

        [Fact]
        public void NormalizesFromFloats()
        {
            float[] data = new float[DataLen];
            int label = m_rand.Next();

            float multBy = m_rand.Next(1024);
            for (int i = 0; i < DataLen; ++i)
            {
                data[i] = (float)m_rand.NextDouble() * multBy;
            }

            IExample example = new NormalizedExample(data, label);

            Assert.StrictEqual(example.Target, label);
            Assert.Equal(example.Input.Min(), 0, 6);
            Assert.Equal(example.Input.Max(), 1, 6);
        }

        [Fact]
        public void NormalizesFromBytes()
        {
            byte[] data = new byte[DataLen];
            int label = m_rand.Next();

            m_rand.NextBytes(data);

            IExample example = new NormalizedExample(data, label);

            Assert.StrictEqual(example.Target, label);
            Assert.Equal(example.Input.Min(), 0, 6);
            Assert.Equal(example.Input.Max(), 1, 6);
        }
    }
}
