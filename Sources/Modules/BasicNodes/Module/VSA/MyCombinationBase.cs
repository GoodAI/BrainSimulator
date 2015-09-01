using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using YAXLib;

namespace GoodAI.Modules.VSA
{
    public enum MyCombinationVector
    {
        Identity,
        ShiftR,
        Involution,

        Random0,
        Random1,
        Random2,
        Random3,
        Random4,
        Random5,
        Random6,
        Random7,
        Random8,
        Random9,
        Random10,
        Random11,
        Random12,
        Random13,
        Random14,
        Random15,
        Random16,
        Random17,
        Random18,
        Random19,
        Random20,
        Random21,
        Random22,
        Random23,
        Random24,
        Random25,
    }


    public abstract class MyCombinationBase : MyRandomPool
    {
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        //Minimal value
        [MyBrowsable, Category("Combinations"), DisplayName("M\tin")]
        [YAXSerializableField(DefaultValue = 0)]
        [Description("Specifies the inclusive lower bound of the generated combination.")]
        public int Min { get; set; }

        //Maximum value
        [MyBrowsable, Category("Combinations")]
        [YAXSerializableField(DefaultValue = 1024)]
        [Description("Specifies the exclusive upper bound of the generated combination.")]
        public int Max { get; set; }


        #region Permutations / Shuffling

        /// <summary>
        /// Shuffles a part of a sequence. The permutation in the segment has only one cycle (there are (n-1)! such permutations).
        /// For more information see http://en.wikipedia.org/wiki/Fisher–Yates_shuffle#Sattolo.27s_algorithm
        /// </summary>
        /// <param name="seq">The segment to be shuffled.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <param name="inPlaceInit">If true, will generate a permutation of numbers from (0, seq.Count - 1). Otherwise, it will permute the original sequence.</param>
        /// <returns>The random permutation.</returns>
        public static void ShuffleSattolo(ArraySegment<float> seq, Random rnd, bool inPlaceInit = true)
        {
            var arr = seq.Array;
            var offset = seq.Offset;

            if (inPlaceInit)
                for (int i = 0; i < seq.Count; i++)
                {
                    int idx = rnd.Next(i) + offset;
                    arr[i + offset] = arr[idx];
                    arr[idx] = i;
                }
            else
                for (int i = 0; i < seq.Count; i++)
                {
                    int idx = rnd.Next(i) + offset;
                    int iidx = i + offset;

                    var tmp = arr[iidx];
                    arr[iidx] = arr[idx];
                    arr[idx] = tmp;
                }
        }

        /// <summary>
        /// Shuffles a part of a sequence. The permutation in the segment has random cycles (there are n! such permutations).
        /// For more information see http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_.22inside-out.22_algorithm
        /// </summary>
        /// <param name="seq">The segment to be shuffled.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <param name="inPlaceInit">If true, will generate a permutation of numbers from (0, seq.Count - 1). Otherwise, it will permute the original sequence.</param>
        /// <returns>The random permutation.</returns>
        public static void ShuffleFisherYates(ArraySegment<float> seq, Random rnd, bool inPlaceInit = true)
        {
            var arr = seq.Array;
            var offset = seq.Offset;

            if (inPlaceInit)
                for (int i = 0; i < seq.Count; i++)
                {
                    int idx = rnd.Next(i + 1) + offset;
                    arr[i + offset] = arr[idx];
                    arr[idx] = i;
                }
            else
                for (int i = 0; i < seq.Count; i++)
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

        #endregion

        #region Combinations

        /// <summary>
        /// Generates unique random numbers in the range [<paramref name="min"/>, <paramref name="max"/>), where min is inclusive and max is exclusive and stores them in <paramref name="arr"/>.
        /// For more information see http://codereview.stackexchange.com/a/61372
        /// </summary>
        /// <param name="arr">The array to store the resulting combinations in. The whole array will be populated.</param>
        /// <param name="candidates">The temporary hash table to store intermediate results in.</param>
        /// <param name="min">The minimum value of a resulting element.</param>
        /// <param name="max">The maximum value of a resulting element.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random sample.</returns>
        public static void GenerateCombinationUnique(ArraySegment<float> arr, HashSet<float> candidates, int min, int max, Random rnd)
        {
            Debug.Assert(arr != null && candidates != null, "Must specify the target array and a temporary hash table object");

            int count = arr.Count;

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
            candidates.CopyTo(arr.Array, arr.Offset);

            // shuffle the results because HashSet has messed
            // with the order, and the algorithm does not produce
            // random-ordered results (e.g. max-1 will never be the first value)
            ShuffleFisherYates(arr, rnd, false);
        }

        /// <summary>
        /// Generates non-unique random numbers in the range [<paramref name="min"/>, <paramref name="max"/>), where min is inclusive and max is exclusive and stores them in <paramref name="arr"/>.
        /// </summary>
        /// <param name="arr">The array to store the resulting combinations in. The whole array will be populated.</param>
        /// <param name="min">The minimum value of a resulting element.</param>
        /// <param name="max">The maximum value of a resulting element.</param>
        /// <param name="rnd">The random number generator used to permute the sequence.</param>
        /// <returns>The random sample.</returns>
        public static void GenerateCombination(ArraySegment<float> arr, int min, int max, Random rnd)
        {
            int count = arr.Count;

            Debug.Assert(!(max <= min
                // max - min > 0 required to avoid overflow
                   || (count > max - min && max - min > 0)),
                // need to use 64-bit to support big ranges (negative min, positive max)
                "Range " + min + " to " + max + " (" + ((Int64)max - (Int64)min) + " values), or count " + count + " is illegal");


            for (int i = arr.Offset; i < arr.Offset + arr.Count; i++)
            {
                arr.Array[i] = rnd.Next(min, max);
            }
        }

        #endregion
    }
}
