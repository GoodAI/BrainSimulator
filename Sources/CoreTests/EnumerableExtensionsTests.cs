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

        [Test]
        [Ignore("Manual")]
        public void EachWithIndexPerformanceTest()
        {
            var data = new[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

            var s1 = Stopwatch.StartNew();
            for (int i = 0; i < 10000000; i++)
            {
                int res = 0;
                for (int j = 0; j < data.Length; j++)
                {
                    res += data[j];
                    res += data[j];
                    res += data[j];
                }
            }
            s1.Stop();

            var s2 = Stopwatch.StartNew();
            for (int i = 0; i < 10000000; i++)
            {
                int res = 0;
                int j = 0;
                foreach (var val in data)
                {
                    res += val;
                    res += val;
                    res += val;
                }
            }
            s2.Stop();

            var s3 = Stopwatch.StartNew();
            for (int i = 0; i < 10000000; i++)
            {
                int res = 0;
                data.EachWithIndex((i1, i2) =>
                {
                    res += i1;
                    res += i1;
                    res += i1;
                });
            }
            s3.Stop();

            Console.WriteLine("{0} vs {1} vs {2}", s1.ElapsedMilliseconds, s2.ElapsedMilliseconds, s3.ElapsedMilliseconds);

            Assert.True(true);
        }
    }
}
