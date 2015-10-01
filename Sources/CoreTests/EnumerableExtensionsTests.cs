using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Platform.Core.Utils;
using Xunit;
using Xunit.Extensions;

namespace CoreTests
{
    public class EnumerableExtensionsTests
    {
        public static TheoryData<ITester> TestCases = new TheoryData<ITester>()
        {
            new Tester<int>(new int[0]),
            new Tester<string>(new [] {"1", "@", "P"}),
            new Tester<StringBuilder>(new [] {new StringBuilder("a"), new StringBuilder("b"), new StringBuilder()})
        };

        [Theory, MemberData("TestCases")]
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

                Assert.Equal(expected, got);
            }

            public override string ToString()
            {
                return "Tester " + m_data;
            }
        }
    }
}
