using System;
using System.Collections.Generic;
using VRage.Collections;
using Xunit;

namespace ToyWorldTests.Utils
{
    public class CircularListTests
    {
        private CircularList<int> m_list;

        public CircularListTests()
        {
            m_list = new CircularList<int>(10);
            m_list.MoveNext();

            for (int i = 0; i < m_list.Size; ++i)
                m_list[i] = i;
        }

        [Fact]
        public void TestSetting()
        {
            Assert.Equal(0, m_list[0]);
            Assert.Equal(4, m_list[4]);
            Assert.Equal(9, m_list[9]);
        }

        [Fact]
        public void TestMovingToNextItem()
        {
            Assert.Equal(2, m_list[2]);
            m_list.MoveNext();
            Assert.Equal(3, m_list[2]);
        }

        [Fact]
        public void TestCurrentSameAsZeroth()
        {
            Assert.Equal(m_list.Current, m_list[0]);
        }

        [Fact]
        public void TestCurrent()
        {
            Assert.Equal(0, m_list.Current);
            m_list.MoveNext();
            Assert.Equal(1, m_list.Current);
        }

        [Fact]
        public void TestCurrentAfterReset()
        {
            m_list.Reset();
            Assert.Throws<ArgumentOutOfRangeException>(() => m_list.Current);
        }

        [Fact]
        public void TestMovingAfterEnd()
        {
            for (int i = 0; i < m_list.Size - 1; ++i)
                m_list.MoveNext();

            Assert.Equal(9, m_list.Current);
            m_list.MoveNext();
            Assert.Equal(0, m_list.Current);
        }

        [Fact]
        public void TestReset()
        {
            m_list.MoveNext();
            Assert.Equal(1, m_list.Current);
            m_list.Reset();
            m_list.MoveNext();
            Assert.Equal(0, m_list.Current);
        }

        [Fact]
        public void TestEnumerator()
        {
            IEnumerator<int> enumerator = m_list.GetEnumerator();
            enumerator.MoveNext();

            Assert.Equal(1, enumerator.Current);
        }
    }
}
