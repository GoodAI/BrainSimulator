using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using GoodAI.LowLevelUtils.IO;

namespace GoodAI.LowLevelsUtilsUnitTests.IO
{
    [TestClass]
    public class FloatBufferReaderTest
    {
        IBufferReader reader;
        float[] buffer;

        [TestInitialize]
        public void SetupBufferAndReader()
        {
            buffer = new float[1];
            reader = BufferReaderFactory.GetFloatReader(buffer);
        }

        public void CheckRounding(float f, int expected)
        {
            buffer[0] = f;
            reader.Buffer = buffer;  // rewind to the start of the buffer

            Assert.AreEqual(expected, reader.ReadInt());
        }

        [TestMethod]
        public void ReadsOneInt()
        {
            CheckRounding(7.15f, 7);
        }

        [TestMethod]
        public void RoundsUpToNearestInt()
        {
            CheckRounding(6.83f, 7);
        }

        [TestMethod]
        public void RoundsHalfUp()  // more precisely "away from zero"
        {
            CheckRounding(4.5f, 5);
            CheckRounding(-2.5f, -3);
        }

        [TestMethod]
        public void ReadsBoolean()
        {
            buffer[0] = 5.0f;

            Assert.AreEqual(true, reader.ReadBool());
        }

        [TestMethod]
        public void ReadsCloseToZeroAsFalse()
        {
            buffer[0] = -0.35f;

            Assert.AreEqual(false, reader.ReadBool());
        }
    }
}
