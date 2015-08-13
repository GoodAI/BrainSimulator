using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using GoodAI.LowLevelUtils.IO;

namespace GoodAI.LowLevelsUtilsUnitTests.IO
{
    [TestClass]
    public class FloatBufferWriterTest
    {
        const double DELTA = 0.000001;

        private float[] m_buffer = new float[100];

        private BufferWriter CreateWriter(float[] buffer)
        {
            return BufferWriterFactory.GetFloatWriter(buffer);
        }

        private BufferWriter CreateWriter()
        {
            return CreateWriter(m_buffer);
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
            var writer = CreateWriter();
            
            writer.PutDouble(77.7);  // should be overwritten

            writer.Buffer = m_buffer;
            writer.PutDouble(5.0);

            Assert.AreEqual(5.0f, m_buffer[0], DELTA);
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
            var writer = CreateWriter();

            writer.PutInt(7);
            writer.PutInt(-1);

            Assert.AreEqual(7.0f, m_buffer[0], DELTA);
            Assert.AreEqual(-1.0f, m_buffer[1], DELTA);
        }

        [TestMethod]
        public void WritesTwoBoolsAndInt()
        {
            var writer = CreateWriter();

            writer.PutBool(true);
            writer.PutBool(false);
            writer.PutInt(7);

            Assert.AreEqual(1.0f, m_buffer[0], DELTA);
            Assert.AreEqual(0.0f, m_buffer[1], DELTA);
            Assert.AreEqual(7.0f, m_buffer[2], DELTA);
        }

        [TestMethod]
        public void GenericFactoryReturnsWorkingWriter()
        {
            var writer = BufferWriterFactory<ConstructableBufferWriter>.GetFloatWriter(m_buffer);

            writer.PutInt(3);
            Assert.AreEqual(3.0f, m_buffer[0]);
        }
    }
}
