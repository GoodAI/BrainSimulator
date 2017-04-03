using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Core.Utils
{
    public static class EnumerableExtensions
    {
        public static void ForEach<T>(this IEnumerable<T> sequence, Action<T> action)
        {
            foreach (var item in sequence)
            {
                action(item);
            }
        }

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

        public static IEnumerable<TResult> SelectManyOrDefault<TSource, TResult>(this IEnumerable<TSource> source, Func<TSource, IEnumerable<TResult>> selector, TResult defaultValue)
        {
            return !source.Any() ? new List<TResult> { defaultValue } : source.SelectMany(selector);
        }
    }
}