using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.ToyWorld.Language
{
    /// The N-best list is a list of N scored items. Adding a new item will 
    /// replace the existing item with the lowest score.
    /// </summary>
    /// <typeparam name="T">The item type</typeparam>
    public abstract class AbstractNBestList<T>
    {
        /// <summary>
        /// The number of items (N) in the N-best list
        /// </summary>
        public int NumberOfItems { get; protected set; }

        /// <summary>
        /// Tests if a score beats the current lowest scorer or the list isn't 
        /// full.
        /// </summary>
        /// <param name="score">A score</param>
        /// <returns>True if better than the current lowest scorer</returns>
        public abstract bool IsBetter(float score);

        /// <summary>
        /// Adds an item without checking the score.
        /// </summary>
        /// <param name="score">The score of the item</param>
        /// <param name="item">The item</param>
        public abstract void Insert(float score, T item);

        /// <summary>
        /// Inserts the item if its score beats the current lowest scorer,
        /// replacing the lowest scorer.
        /// </summary>
        /// <param name="score">The score of the item</param>
        /// <param name="item">The item</param>
        public abstract void InsertIfBetter(float score, T item);

        /// <summary>
        /// Returns a list of scores and items, sorted by descending score
        /// </summary>
        /// <returns>Sorted list of scored items</returns>
        public abstract List<Tuple<float, T>> GetSortedList();
    }

    public class NBestComparer<T> : Comparer<Tuple<float, T>>
    {
        public override int Compare(Tuple<float, T> x, Tuple<float, T> y)
        {
            return x.Item1.CompareTo(y.Item1);
        }
    }

    public class NBiggestComparer<T> : Comparer<Tuple<float, T>>
    {
        public override int Compare(Tuple<float, T> x, Tuple<float, T> y)
        {
            return x.Item1.CompareTo(y.Item1);
        }
    }

    public class NSmallestComparer<T> : Comparer<Tuple<float, T>>
    {
        public override int Compare(Tuple<float, T> x, Tuple<float, T> y)
        {
            return y.Item1.CompareTo(x.Item1);
        }
    }

    /// <summary>
    /// The concrete N-best list is implemented using a heap.
    /// </summary>
    /// <typeparam name="T">The item type</typeparam>
    public class NBestList<T> : AbstractNBestList<T>
    {
        /// <summary>
        /// The heap that tracks the elements in the N-best list
        /// </summary>
        private MinHeap<Tuple<float, T>> heap = new MinHeap<Tuple<float, T>>(new NBestComparer<T>());

        /// <summary>
        /// Constructs the N-best list
        /// </summary>
        /// <param name="numberOfItems">The number of items (N) in the list</param>
        public NBestList(int numberOfItems)
        {
            NumberOfItems = numberOfItems;
        }

        public override bool IsBetter(float score)
        {
            // TODO use comparer
            return heap.Count() < NumberOfItems || heap.GetMin().Item1 < score;
        }

        public override void Insert(float score, T item)
        {
            heap.Add(new Tuple<float, T>(score, item));
            if (heap.Count() > NumberOfItems)
            {
                heap.ExtractDominating();
            }
        }

        public override void InsertIfBetter(float score, T item)
        {
            if (IsBetter(score))
            {
                Insert(score, item);
            }
        }

        public override List<Tuple<float, T>> GetSortedList()
        {
            var list = new List<Tuple<float, T>>();
            foreach (var tuple in heap)
            {
                list.Add(tuple);
            }
            return list.OrderByDescending(a => a, new NBestComparer<T>()).ToList();
        }
    }
}
