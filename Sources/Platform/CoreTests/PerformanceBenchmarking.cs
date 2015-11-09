using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Platform.Core.Utils;
using Xunit;

namespace CoreTests
{
    public class PerformanceBenchmarking
    {
        private class Parent
        {
            public int Foo()
            {
                return 1;
            }
        }

        private class Child : Parent
        {
            public int Bar()
            {
                return 2;
            }
        }

        // [Fact, Trait("Category", "Manual")] NOTE: just a performance test, uncomment when needed
        public void EachWithIndexPerformanceTest()
        {
            const int IterationCount = 1000000;  // increase for greater precision

            var data = new[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

            var s1 = Stopwatch.StartNew();
            for (int i = 0; i < IterationCount; i++)
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
            for (int i = 0; i < IterationCount; i++)
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
            for (int i = 0; i < IterationCount; i++)
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

            Console.WriteLine("for loop {0} vs foreach {1} vs EachWithIndex {2}", s1.ElapsedMilliseconds, s2.ElapsedMilliseconds, s3.ElapsedMilliseconds);

            Assert.True(true);
        }
    }
}
