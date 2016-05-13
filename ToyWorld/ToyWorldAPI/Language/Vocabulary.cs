using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.ToyWorld.Language
{
    /// <summary>
    /// Words and word vectors.
    /// The Vocabulary is a singleton, implemented using the recipe at
    /// http://csharpindepth.com/Articles/General/Singleton.aspx#lazy
    /// </summary>
    public sealed class Vocabulary
    {
        /// <summary>
        /// The number of dimension of the vector space
        /// </summary>
        public int NumberOfDimensions { get; set; }

        /// <summary>
        /// Dictionary for label-to-vector lookup
        /// </summary>
        private Dictionary<string, float[]> labeledVectorDictionary = null;

        /// <summary>
        /// Random number generator for making random vectors
        /// </summary>
        private Random rnd = new Random();

        /// <summary>
        /// Lazy singleton instantiation
        /// </summary>
        private static readonly Lazy<Vocabulary> lazy = new Lazy<Vocabulary>(() => new Vocabulary());
 
        /// <summary>
        /// Singleton instance
        /// </summary>
        public static Vocabulary Instance { get { return lazy.Value; } }

        /// <summary>
        /// Private constructor for the singleton
        /// </summary>
        private Vocabulary()
        {
        }

        /// <summary>
        /// Initializes (or resets) the vocabulary.
        /// </summary>
        /// <param name="vectorDimensions">The number of word vector dimensions</param>
        public void Initialize(int vectorDimensions)
        {
            NumberOfDimensions = vectorDimensions;
            labeledVectorDictionary = new Dictionary<string, float[]>();
        }

        /// <summary>
        /// Adds a labeled vector to the vocabulary.
        /// </summary>
        /// <param name="label">The label</param>
        /// <param name="vector">The vector</param>
        public void Add(string label, float[] vector)
        {
            labeledVectorDictionary.Add(label, vector);
        }

        /// <summary>
        /// Retrieves the vector that corresponds to a label.
        /// </summary>
        /// <param name="label">The label to look up</param>
        /// <returns>The vector corresponding to the label</returns>
        public float[] VectorFromLabel(string label)
        {
            string normalizedLabel = label.ToLowerInvariant();
            float[] vector = null;
            bool hasLabel = labeledVectorDictionary.TryGetValue(normalizedLabel, out vector);
            return hasLabel ? vector : GetOOVVector(normalizedLabel);
        }

        /// <summary>
        /// Returns a vector for an out-of-vocabulary word.
        /// This is currently done by drawing a new random vector and assigning it to the word.
        /// </summary>
        /// <returns>Word vector</returns>
        private float[] GetOOVVector(string label)
        {
            float[] newVector = MakeRandomVector();
            Add(label, newVector);
            return newVector;
        }

        /// <summary>
        /// Create a random vector uniformly distributed over the vector space.
        /// Components are in [0, 1[.
        /// </summary>
        /// <returns>Random vector</returns>
        private float[] MakeRandomVector()
        {
            float[] vector = new float[NumberOfDimensions];
            for (int elementIndex = 0; elementIndex < NumberOfDimensions; elementIndex++)
            {
                vector[elementIndex] = (float)rnd.NextDouble();
            }
            return vector;
        }

        /// <summary>
        /// Returns true if the argument is the zero vector.
        /// </summary>
        /// <param name="vector">The input vector</param>
        /// <returns>True if all elements are zero</returns>
        public static bool IsZero(float[] vector)
        {
            foreach (float element in vector)
            {
                if (element != 0)
                {
                    return false;
                }
            }
            return true;
        }

        /// <summary>
        /// Finds the vectors most similar to the input vector. If the input is
        /// the zero vector, an empty list is returned.
        /// </summary>
        /// <param name="vector">The input vector</param>
        /// <param name="neighborhoodSize">The number of neighbors to retrieve</param>
        /// <returns>A list of neighboring vectors sorted by descending cosine similarity</returns>
        public List<Tuple<float, LabeledVector>> FindNearestNeighbors(LabeledVector vector, int neighborhoodSize)
        {
            return FindNearestNeighbors(vector.Vector, neighborhoodSize);
        }

        /// <summary>
        /// Finds the vectors most similar to the input vector. If the input is
        /// the zero vector, an empty list is returned.
        /// </summary>
        /// <param name="vector">The input vector</param>
        /// <param name="neighborhoodSize">The number of neighbors to retrieve</param>
        /// <returns>A list of neighboring vectors sorted by descending cosine similarity</returns>
        public List<Tuple<float, LabeledVector>> FindNearestNeighbors(float[] vector, int neighborhoodSize)
        {
            var nBestList = new NBestList<LabeledVector>(neighborhoodSize);
            if (!IsZero(vector))
            {
                foreach (var wordVectorPair in labeledVectorDictionary)
                {
                    float cosine = LabeledVector.Cosine(wordVectorPair.Value, vector);
                    if (nBestList.IsBetter(cosine))
                    {
                        nBestList.Insert(cosine, new LabeledVector(wordVectorPair.Key, wordVectorPair.Value));
                    }
                }
            }
            return nBestList.GetSortedList();
        }

    }
}
