using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Platform.Core.Utils
{
    public static class EnumerableExtensions
    {
        public static void EachWithIndex<T>(this IEnumerable<T> input, Action<T, int> action)
        {
            int i = 0;
            foreach (var item in input)
            {
                action(item, i);
                i++;
            }
        }

        public class Indexed<T>
        {
            public T Item { get; set; }
            public int Index { get; set; }

            public Indexed(T item, int index)
            {
                Item = item;
                Index = index;
            }
        }

        public static IEnumerable<Indexed<T>> WithIndex<T>(this IEnumerable<T> input)
        {
            var i = 0;
            foreach (T item in input)
            {
                yield return new Indexed<T>(item, i);
                i++;
            }
        }
    }
}