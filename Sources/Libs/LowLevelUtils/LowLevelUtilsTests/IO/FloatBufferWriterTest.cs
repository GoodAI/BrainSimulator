using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using GoodAI.LowLevelUtils.IO;

namespace GoodAI.LowLevelsUtilsUnitTests.IO
{
    [TestClass]
    public class FloatBufferWriterTest
    {
        const double DELTA = 0.000001;

        private BufferWriter CreateWriter(float[] buffer)
        {
            return BufferWriterFactory.GetFloatWriter(buffer);
        }

        [TestMethod]
        public void WritesDouble()
        {
            var floatBuffer = new float[1];
            var writer = CreateWriter(floatBuffer);

            writer.PutDouble(5.0);

            Assert.AreEqual(5.0f, floatBuffer[0], DELTA);
        }

        [TestMethod]
        public void WritesTwoDoubles()
        {
            var floatBuffer = new float[2];
            var writer = CreateWriter(floatBuffer);

            writer.PutDouble(0.0);
            writer.PutDouble(1.0);

            Assert.AreEqual(0.0f, floatBuffer[0], DELTA);
            Assert.AreEqual(1.0f, floatBuffer[1], DELTA);
        }

        [TestMethod]
        public void WritesFromStartAfterSetBuffer()
        {
            var floatBuffer = new float[5];
            var writer = CreateWriter(floatBuffer);
            
            writer.PutDouble(77.7);  // should be overwritten

            writer.Buffer = floatBuffer;
            writer.PutDouble(5.0);

            Assert.AreEqual(5.0f, floatBuffer[0], DELTA);
        }

        [TestMethod, ExpectedException(typeof(InvalidOperationException))]
        public void ThrowsInvalidOperationWithoutBuffer()
        {
            var writer = BufferWriterFactory.GetFloatWriter();

            writer.PutDouble(1.0);
        }

        [TestMethod]
        public void WritesTwoInts()
        {
            var floatBuffer = new float[2];
            var writer = CreateWriter(floatBuffer);

            writer.PutInt(7);
            writer.PutInt(-1);

            Assert.AreEqual(7.0f, floatBuffer[0], DELTA);
            Assert.AreEqual(-1.0f, floatBuffer[1], DELTA);
        }

        [TestMethod]
        public void WritesTwoBoolsAndInt()
        {
            var floatBuffer = new float[3];
            var writer = CreateWriter(floatBuffer);

            writer.PutBool(true);
            writer.PutBool(false);
            writer.PutInt(7);

            Assert.AreEqual(1.0f, floatBuffer[0], DELTA);
            Assert.AreEqual(0.0f, floatBuffer[1], DELTA);
            Assert.AreEqual(7.0f, floatBuffer[2], DELTA);
        }
    }
}
