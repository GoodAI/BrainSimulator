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

        [Fact, Trait("Category", "Manual")]
        public void TestIsAs()
        {
            var list = new List<Parent>();
            for (int i = 0; i < 10000000; i++)
            {
                list.Add(i % 2 == 0 ? new Parent() : new Child());
            }

            var s1 = Stopwatch.StartNew();
            foreach (var item in list)
            {
                if (item is Child)
                {
                    (item as Child).Bar();
                }
                else
                {
                    item.Foo();
                }
            }
            s1.Stop();

            var s2 = Stopwatch.StartNew();
            foreach (var item in list)
            {
                var childItem = item as Child;
                if (childItem != null)
                {
                    childItem.Bar();
                }
                else
                {
                    item.Foo();
                }
            }
            s2.Stop();

            Console.WriteLine(string.Format("is and as: {0} versus as and null-check: {1}", s1.ElapsedMilliseconds, s2.ElapsedMilliseconds));
        }

        [Fact, Trait("Category", "Manual")]
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

            Console.WriteLine("for loop {0} vs foreach {1} vs EachWithIndex {2}", s1.ElapsedMilliseconds, s2.ElapsedMilliseconds, s3.ElapsedMilliseconds);

            Assert.True(true);
        }
    }
}
