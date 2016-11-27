using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using VRage.Extensions;

namespace System
{
    public static class ArrayExtensions
    {
        public static bool IsValidIndex<T>(this T[] self, int index)
        {
            // Only one unsigned comparison instead of signed comparison against 0 and length.
            return ((uint)index < (uint)self.Length);
        }

        public static bool IsNullOrEmpty<T>(this T[] self)
        {
            return self == null || self.Length == 0;
        }

        public static bool TryGetValue<T>(this T[] self, int index, out T value)
        {
            if ((uint)index < (uint)self.Length)
            {
                value = self[index];
                return true;
            }
            else
            {
                value = default(T);
                return false;
            }
        }

        public static T[] RemoveIndices<T>(this T[] self, List<int> indices)
        {
            if (indices.Count >= self.Length)
                return new T[0];

            if (indices.Count == 0)
                return self;

            T[] better = new T[self.Length - indices.Count];
            int offset = 0;
            for (int i = 0; i < self.Length - indices.Count; i++)
            {
                while (offset < indices.Count && i == indices[offset] - offset)
                    offset++;
                better[i] = self[i + offset];
            }

            return better;
        }

        /**
         * Do a binary search in an array of interval limits, each member is the interval threshold.
         * 
         * The result is the index of the interval that contains the value searched for.
         * 
         * If the interval array is empty 0 is returned (as we assume we have only the (-∞,+∞) interval).
         * 
         * Return range: [0, Length]
         */
        public static int BinaryIntervalSearch<T>(this T[] self, T value) where T : IComparable<T>
        {
            if (self.Length == 0) return 0;
            if (self.Length == 1)
            {
                return value.CompareTo(self[0]) > 0 ? 1 : 0;
            }

            int mid;
            int start = 0, end = self.Length;

            while (end - start > 1)
            {
                mid = (start + end) / 2;

                if (value.CompareTo(self[mid]) > 0)
                {
                    start = mid;
                }
                else
                {
                    end = mid;
                }
            }

            int ret = start;

            // end of array;
            if (value.CompareTo(self[start]) > 0)
            {
                ret = end;
            }

            return ret;
        }

        /// <summary>
        /// OfType on array implemented without allocations
        /// </summary>
        public static ArrayOfTypeEnumerator<TBase, ArrayEnumerator<TBase>, T> OfTypeFast<TBase, T>(this TBase[] array)
            where T : TBase
        {
            return new ArrayOfTypeEnumerator<TBase, ArrayEnumerator<TBase>, T>(new ArrayEnumerator<TBase>(array));
        }


        #region Permutations / Shuffling

        #region Sattolo

        /// <summary>
        /// Shuffles a part of a sequence. The permutation in the segment has only one cycle (there are (n-1)! such permutations).
        /// For more information see http://en.wikipedia.org/wiki/Fisher–Yates_shuffle#Sattolo.27s_algorithm
        /// </summary>
        /// <param name="arr">The segment to be shuffled.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random permutation.</returns>
        public static void ShuffleSattolo<T>(this T[] arr, Random rnd)
        {
            ShuffleSattolo(new ArraySegment<T>(arr), rnd);
        }

        /// <summary>
        /// Shuffles a part of a sequence. The permutation in the segment has only one cycle (there are (n-1)! such permutations).
        /// For more information see http://en.wikipedia.org/wiki/Fisher–Yates_shuffle#Sattolo.27s_algorithm
        /// </summary>
        /// <param name="seg">The segment to be shuffled.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random permutation.</returns>
        public static void ShuffleSattolo<T>(this ArraySegment<T> seg, Random rnd)
        {
            var arr = seg.Array;
            var offset = seg.Offset;

            for (int i = 0; i < seg.Count; i++)
            {
                int idx = rnd.Next(i) + offset;
                int iidx = i + offset;

                var tmp = arr[iidx];
                arr[iidx] = arr[idx];
                arr[idx] = tmp;
            }
        }


        /// <summary>
        /// Shuffles a part of a sequence. The permutation in the segment has only one cycle (there are (n-1)! such permutations).
        /// For more information see http://en.wikipedia.org/wiki/Fisher–Yates_shuffle#Sattolo.27s_algorithm
        /// </summary>
        /// <param name="arr">The segment to be shuffled.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random permutation.</returns>
        public static void ShuffleSattoloInPlace(this int[] arr, Random rnd)
        {
            ShuffleSattoloInPlace(new ArraySegment<int>(arr), rnd);
        }

        /// <summary>
        /// Shuffles a part of a sequence. The permutation in the segment has only one cycle (there are (n-1)! such permutations).
        /// For more information see http://en.wikipedia.org/wiki/Fisher–Yates_shuffle#Sattolo.27s_algorithm
        /// </summary>
        /// <param name="arr">The segment to be shuffled.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random permutation.</returns>
        public static void ShuffleSattoloInPlace(this float[] arr, Random rnd)
        {
            ShuffleSattoloInPlace(new ArraySegment<float>(arr), rnd);
        }

        /// Will generate a permutation of numbers from (0, seq.Count - 1).
        public static void ShuffleSattoloInPlace(this ArraySegment<int> seq, Random rnd)
        {
            var arr = seq.Array;
            var offset = seq.Offset;

            for (int i = 0; i < seq.Count; i++)
            {
                int idx = rnd.Next(i) + offset;
                arr[i + offset] = arr[idx];
                arr[idx] = i;
            }
        }

        /// Will generate a permutation of numbers from (0, seq.Count - 1).
        public static void ShuffleSattoloInPlace(this ArraySegment<float> seq, Random rnd)
        {
            var arr = seq.Array;
            var offset = seq.Offset;

            for (int i = 0; i < seq.Count; i++)
            {
                int idx = rnd.Next(i) + offset;
                arr[i + offset] = arr[idx];
                arr[idx] = i;
            }
        }

        #endregion

        #region FisherYates

        /// <summary>
        /// Shuffles a part of a sequence. The permutation in the segment has random cycles (there are n! such permutations).
        /// For more information see http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_.22inside-out.22_algorithm
        /// </summary>
        /// <param name="arr">The segment to be shuffled.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random permutation.</returns>
        public static void ShuffleFisherYates<T>(this T[] arr, Random rnd)
        {
            ShuffleFisherYates(new ArraySegment<T>(arr), rnd);
        }

        /// <summary>
        /// Shuffles a part of a sequence. The permutation in the segment has random cycles (there are n! such permutations).
        /// For more information see http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_.22inside-out.22_algorithm
        /// </summary>
        /// <param name="seg">The segment to be shuffled.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random permutation.</returns>
        public static void ShuffleFisherYates<T>(this ArraySegment<T> seg, Random rnd)
        {
            var arr = seg.Array;
            var offset = seg.Offset;

            for (int i = 0; i < seg.Count; i++)
            {
                int idx = rnd.Next(i + 1) + offset;
                int iidx = i + offset;

                //if (idx == iidx)
                //    continue;

                var tmp = arr[iidx];
                arr[iidx] = arr[idx];
                arr[idx] = tmp;
            }
        }


        /// <summary>
        /// Shuffles a part of a sequence. The permutation in the segment has random cycles (there are n! such permutations).
        /// For more information see http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_.22inside-out.22_algorithm
        /// </summary>
        /// <param name="arr">The segment to be shuffled.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random permutation.</returns>
        public static void ShuffleFisherYatesInPlace(this int[] arr, Random rnd)
        {
            ShuffleFisherYatesInPlace(new ArraySegment<int>(arr), rnd);
        }

        /// <summary>
        /// Shuffles a part of a sequence. The permutation in the segment has random cycles (there are n! such permutations).
        /// For more information see http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_.22inside-out.22_algorithm
        /// </summary>
        /// <param name="arr">The segment to be shuffled.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random permutation.</returns>
        public static void ShuffleFisherYatesInPlace(this float[] arr, Random rnd)
        {
            ShuffleFisherYatesInPlace(new ArraySegment<float>(arr), rnd);
        }

        /// Will generate a permutation of numbers from (0, seq.Count - 1).
        public static void ShuffleFisherYatesInPlace(this ArraySegment<int> seq, Random rnd)
        {
            var arr = seq.Array;
            var offset = seq.Offset;

            for (int i = 0; i < seq.Count; i++)
            {
                int idx = rnd.Next(i + 1) + offset;
                arr[i + offset] = arr[idx];
                arr[idx] = i;
            }
        }

        /// Will generate a permutation of numbers from (0, seq.Count - 1).
        public static void ShuffleFisherYatesInPlace(this ArraySegment<float> seq, Random rnd)
        {
            var arr = seq.Array;
            var offset = seq.Offset;

            for (int i = 0; i < seq.Count; i++)
            {
                int idx = rnd.Next(i + 1) + offset;
                arr[i + offset] = arr[idx];
                arr[idx] = i;
            }
        }

        #endregion

        #endregion

        #region Combinations

        #region Without repetition

        /// <summary>
        /// Generates unique random integers in the range [<paramref name="min"/>, <paramref name="max"/>), where min is inclusive and max is exclusive and stores them in <paramref name="seg"/>.
        /// For more information see http://codereview.stackexchange.com/a/61372
        /// </summary>
        /// <param name="arr">The array to store the resulting combinations in. The whole array will be populated.</param>
        /// <param name="candidates">The temporary hash table to store intermediate results in.</param>
        /// <param name="min">The minimum value of a resulting element.</param>
        /// <param name="max">The maximum value of a resulting element.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random sample.</returns>
        public static void GenerateCombinationUnique(this int[] arr, HashSet<int> candidates, int min, int max, Random rnd)
        {
            GenerateCombinationUnique(new ArraySegment<int>(arr), candidates, min, max, rnd);
        }

        /// <summary>
        /// Generates unique random integers in the range [<paramref name="min"/>, <paramref name="max"/>), where min is inclusive and max is exclusive and stores them in <paramref name="seg"/>.
        /// For more information see http://codereview.stackexchange.com/a/61372
        /// </summary>
        /// <param name="arr">The array to store the resulting combinations in. The whole array will be populated.</param>
        /// <param name="candidates">The temporary hash table to store intermediate results in.</param>
        /// <param name="min">The minimum value of a resulting element.</param>
        /// <param name="max">The maximum value of a resulting element.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random sample.</returns>
        public static void GenerateCombinationUnique(this float[] arr, HashSet<float> candidates, int min, int max, Random rnd)
        {
            GenerateCombinationUnique(new ArraySegment<float>(arr), candidates, min, max, rnd);
        }

        /// <summary>
        /// Generates unique random integers in the range [<paramref name="min"/>, <paramref name="max"/>), where min is inclusive and max is exclusive and stores them in <paramref name="seg"/>.
        /// For more information see http://codereview.stackexchange.com/a/61372
        /// </summary>
        /// <param name="seg">The array to store the resulting combinations in. The whole array will be populated.</param>
        /// <param name="candidates">The temporary hash table to store intermediate results in.</param>
        /// <param name="min">The minimum value of a resulting element.</param>
        /// <param name="max">The maximum value of a resulting element.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random sample.</returns>
        public static void GenerateCombinationUnique(this ArraySegment<int> seg, HashSet<int> candidates, int min, int max, Random rnd)
        {
            Debug.Assert(seg != null && candidates != null, "Must specify the target array and a temporary hash table object");

            int count = seg.Count;

            Debug.Assert(!(max <= min
                // max - min > 0 required to avoid overflow
                   || (count > max - min && max - min > 0)),
                // need to use 64-bit to support big ranges (negative min, positive max)
                "Range " + min + " to " + max + " (" + ((Int64)max - (Int64)min) + " values), or count " + count + " is illegal");


            // generate count random values.
            candidates.Clear();

            // start count values before max, and end at max
            for (int top = max - count; top < max; top++)
            {
                // May strike a duplicate.
                // Need to add +1 to make inclusive generator
                // +1 is safe even for MaxVal max value because top < max
                if (!candidates.Add(rnd.Next(min, top + 1)))
                {
                    // collision, add inclusive max.
                    // which could not possibly have been added before.
                    candidates.Add(top);
                }
            }

            // load them in to a list, to sort
            candidates.CopyTo(seg.Array, seg.Offset);

            // shuffle the results because HashSet has messed
            // with the order, and the algorithm does not produce
            // random-ordered results (e.g. max-1 will never be the first value)
            ShuffleFisherYatesInPlace(seg, rnd);
        }

        /// <summary>
        /// Generates unique random integers in the range [<paramref name="min"/>, <paramref name="max"/>), where min is inclusive and max is exclusive and stores them in <paramref name="seg"/>.
        /// For more information see http://codereview.stackexchange.com/a/61372
        /// </summary>
        /// <param name="seg">The array to store the resulting combinations in. The whole array will be populated.</param>
        /// <param name="candidates">The temporary hash table to store intermediate results in.</param>
        /// <param name="min">The minimum value of a resulting element.</param>
        /// <param name="max">The maximum value of a resulting element.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random sample.</returns>
        public static void GenerateCombinationUnique(this ArraySegment<float> seg, HashSet<float> candidates, int min, int max, Random rnd)
        {
            Debug.Assert(seg != null && candidates != null, "Must specify the target array and a temporary hash table object");

            int count = seg.Count;

            Debug.Assert(!(max <= min
                // max - min > 0 required to avoid overflow
                   || (count > max - min && max - min > 0)),
                // need to use 64-bit to support big ranges (negative min, positive max)
                "Range " + min + " to " + max + " (" + ((Int64)max - (Int64)min) + " values), or count " + count + " is illegal");


            // generate count random values.
            candidates.Clear();

            // start count values before max, and end at max
            for (int top = max - count; top < max; top++)
            {
                // May strike a duplicate.
                // Need to add +1 to make inclusive generator
                // +1 is safe even for MaxVal max value because top < max
                if (!candidates.Add(rnd.Next(min, top + 1)))
                {
                    // collision, add inclusive max.
                    // which could not possibly have been added before.
                    candidates.Add(top);
                }
            }

            // load them in to a list, to sort
            candidates.CopyTo(seg.Array, seg.Offset);

            // shuffle the results because HashSet has messed
            // with the order, and the algorithm does not produce
            // random-ordered results (e.g. max-1 will never be the first value)
            ShuffleFisherYatesInPlace(seg, rnd);
        }

        #endregion

        #region With repetition

        /// <summary>
        /// Generates unique random integers in the range [<paramref name="min"/>, <paramref name="max"/>), where min is inclusive and max is exclusive and stores them in <paramref name="seg"/>.
        /// For more information see http://codereview.stackexchange.com/a/61372
        /// </summary>
        /// <param name="arr">The array to store the resulting combinations in. The whole array will be populated.</param>
        /// <param name="min">The minimum value of a resulting element.</param>
        /// <param name="max">The maximum value of a resulting element.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random sample.</returns>
        public static void GenerateCombination(this int[] arr, int min, int max, Random rnd)
        {
            GenerateCombination(new ArraySegment<int>(arr), min, max, rnd);
        }

        /// <summary>
        /// Generates unique random integers in the range [<paramref name="min"/>, <paramref name="max"/>), where min is inclusive and max is exclusive and stores them in <paramref name="seg"/>.
        /// For more information see http://codereview.stackexchange.com/a/61372
        /// </summary>
        /// <param name="arr">The array to store the resulting combinations in. The whole array will be populated.</param>
        /// <param name="min">The minimum value of a resulting element.</param>
        /// <param name="max">The maximum value of a resulting element.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random sample.</returns>
        public static void GenerateCombination(this float[] arr, int min, int max, Random rnd)
        {
            GenerateCombination(new ArraySegment<float>(arr), min, max, rnd);
        }

        /// <summary>
        /// Generates non-unique random numbers in the range [<paramref name="min"/>, <paramref name="max"/>), where min is inclusive and max is exclusive and stores them in <paramref name="seg"/>.
        /// </summary>
        /// <param name="seg">The array to store the resulting combinations in. The whole array will be populated.</param>
        /// <param name="min">The minimum value of a resulting element.</param>
        /// <param name="max">The maximum value of a resulting element.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random sample.</returns>
        public static void GenerateCombination(this ArraySegment<int> seg, int min, int max, Random rnd)
        {
            int count = seg.Count;

            Debug.Assert(!(max <= min
                // max - min > 0 required to avoid overflow
                   || (count > max - min && max - min > 0)),
                // need to use 64-bit to support big ranges (negative min, positive max)
                "Range " + min + " to " + max + " (" + ((Int64)max - (Int64)min) + " values), or count " + count + " is illegal");


            for (int i = seg.Offset; i < seg.Offset + seg.Count; i++)
            {
                seg.Array[i] = rnd.Next(min, max);
            }
        }

        /// <summary>
        /// Generates non-unique random numbers in the range [<paramref name="min"/>, <paramref name="max"/>), where min is inclusive and max is exclusive and stores them in <paramref name="seg"/>.
        /// </summary>
        /// <param name="seg">The array to store the resulting combinations in. The whole array will be populated.</param>
        /// <param name="min">The minimum value of a resulting element.</param>
        /// <param name="max">The maximum value of a resulting element.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random sample.</returns>
        public static void GenerateCombination(this ArraySegment<float> seg, int min, int max, Random rnd)
        {
            int count = seg.Count;

            Debug.Assert(!(max <= min
                // max - min > 0 required to avoid overflow
                   || (count > max - min && max - min > 0)),
                // need to use 64-bit to support big ranges (negative min, positive max)
                "Range " + min + " to " + max + " (" + ((Int64)max - (Int64)min) + " values), or count " + count + " is illegal");


            for (int i = seg.Offset; i < seg.Offset + seg.Count; i++)
            {
                seg.Array[i] = rnd.Next(min, max);
            }
        }

        #endregion

        #endregion
    }
}
