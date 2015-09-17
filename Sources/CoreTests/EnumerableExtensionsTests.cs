using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Platform.Core.Utils;
using NUnit.Framework;

namespace CoreTests
{
    [TestFixture]
    public class EnumerableExtensionsTests
    {
        private IEnumerable<ITester> TestCases()
        {
            yield return new Tester<int>(new int[0]);
            yield return new Tester<string>(new [] {"1", "@", "P"});
        }
        
        [TestCaseSource("TestCases")]
        public void EachWithIndexTest(ITester tester)
        {
            tester.TestEquality();
        }

        public interface ITester
        {
            void TestEquality();
        }

        private class Tester<T> : ITester
        {
            private readonly T[] m_data;

            public Tester(T[] data)
            {
                m_data = data;
            }

            public void TestEquality()
            {
                var data = m_data.ToList();
                var expected = new List<Tuple<int, T>>();

                // ReSharper disable once LoopCanBeConvertedToQuery
                for (int i = 0; i < data.Count; i++)
                    expected.Add(new Tuple<int, T>(i, data[i]));

                var got = new List<Tuple<int, T>>();
                data.EachWithIndex((arg1, i) => got.Add(new Tuple<int, T>(i, arg1)));

                Assert.AreEqual(expected, got);
            }
        }
    }
}
