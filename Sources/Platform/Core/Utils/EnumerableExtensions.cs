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
    }
}
