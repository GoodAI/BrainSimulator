using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Utils;

namespace GoodAI.ToyWorld.Language
{
    /// <summary>
    /// Words and word vectors.
    /// </summary>
    public sealed class Vocabulary
    {
        /// <summary>
        /// The vector type describes the vectors and decides how out-of-vocabulary
        /// words are handled.
        /// </summary>
        public enum WordVectorType
        {
            /// <summary>
            /// Random points in hypercube
            /// </summary>
            Random, 

            /// <summary>
            /// Random points on hypersphere
            /// </summary>
            RandomNormalized, 

            /// <summary>
            /// One-of-N encoding
            /// </summary>
            OneOfN, 

            /// <summary>
            /// Embeddings learned by training on a corpus
            /// </summary>
            Learned 
        }

        /// <summary>
        /// The vector type used (see WordVectorType)
        /// </summary>
        public WordVectorType VectorType = WordVectorType.Random;

        /// <summary>
        /// The number of dimension of the vector space
        /// </summary>
        public int NumberOfDimensions { get; set; }

        /// <summary>
        /// Only allow one thread at a time to create a new vector
        /// </summary>
        private Object _newVectorLock = new Object();

        /// <summary>
        /// The vocabulary size
        /// </summary>
        public int Size
        {
            get { return _labeledVectorDictionary.Count; }
        }

        /// <summary>
        /// Dictionary for label-to-vector lookup
        /// </summary>
        private Dictionary<string, float[]> _labeledVectorDictionary = null;

        /// <summary>
        /// Random number generator for making random vectors
        /// </summary>
        private Random rnd = new Random();

        /// <summary>
        /// Constructs the Vocabulary.
        /// </summary>
        /// <param name="vectorDimensions">The number of word vector dimensions</param>
        /// <param name="vectorType">The vector type used</param>
        public Vocabulary(
            int vectorDimensions,
            WordVectorType vectorType = WordVectorType.Random)
        {
            NumberOfDimensions = vectorDimensions;
            VectorType = vectorType;
            _labeledVectorDictionary = new Dictionary<string, float[]>();            
        }

        /// <summary>
        /// Adds a labeled vector to the vocabulary.
        /// </summary>
        /// <param name="label">The label</param>
        /// <param name="vector">The vector</param>
        public void Add(string label, float[] vector)
        {
            _labeledVectorDictionary.Add(label, vector);
        }

        /// <summary>
        /// Adds a labeled vector to the vocabulary, assigning a default (random) value
        /// to the vector.
        /// </summary>
        /// <param name="label">The label</param>
        public void Add(string label)
        {
            Add(label, MakeNewVector());
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
            bool hasLabel = _labeledVectorDictionary.TryGetValue(normalizedLabel, out vector);
            return hasLabel ? vector : GetOOVVector(normalizedLabel);
        }

        /// <summary>
        /// Returns a new vector for an out-of-vocabulary word.
        /// </summary>
        /// <returns>Word vector</returns>
        private float[] GetOOVVector(string label)
        {
            float[] newVector = MakeNewVector();
            Add(label, newVector);
            return newVector;
        }

        /// <summary>
        /// Creates a new vector.
        /// </summary>
        /// <returns>The new vector</returns>
        private float[] MakeNewVector()
        {
            float[] newVector = null;
            lock (_newVectorLock)
            { 
                switch (VectorType)
                {
                    case WordVectorType.Random:
                        newVector = MakeRandomVector();
                        break;
                    case WordVectorType.OneOfN:
                        newVector = MakeOneOfNVector();
                        break;
                }
            }
            return newVector;           
        }

        /// <summary>
        /// Creates a random vector uniformly distributed over the vector space.
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
        /// Creates a new one-of-N vector.
        /// </summary>
        /// <returns>One-of-N vector</returns>
        private float[] MakeOneOfNVector()
        {
            int newWordIndex = Size;
            float[] vector = new float[NumberOfDimensions];
            for (int elementIndex = 0; elementIndex < NumberOfDimensions; elementIndex++)
            {
                vector[elementIndex] = elementIndex == newWordIndex ? 1 : 0;
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
            if (IsZero(vector))
            {
                return null;
            }

            var nBestList = new NBestList<LabeledVector>(neighborhoodSize);
            foreach (var wordVectorPair in _labeledVectorDictionary)
            {
                float cosine = LabeledVector.Cosine(wordVectorPair.Value, vector);
                if (nBestList.IsBetter(cosine))
                {
                    nBestList.Insert(cosine, new LabeledVector(wordVectorPair.Key, wordVectorPair.Value));
                }
            }
            return nBestList.GetSortedList();
        }

        /// <summary>
        /// Loads the vocabulary from a file in text format (e.g., word2vec)
        /// </summary>
        /// <param name="path">The path to the vocabulary text file</param>
        public void ReadTextFile(string path)
        {
            StreamReader vocabularyReader = File.OpenText(path);
            Read(vocabularyReader);
        }

        /// <summary>
        /// Loads the vocabulary from text using a StreamReader
        /// </summary>
        /// <param name="vocabularyReader">The reader</param>
        public void Read(StreamReader vocabularyReader)
        {
            MyLog.INFO.WriteLine("Loading vector space...");
            ReadFileHeader(vocabularyReader);
            ReadLabeledVectors(vocabularyReader);

            vocabularyReader.Close();
            MyLog.INFO.WriteLine("Done loading vector space.");  
        }

        /// <summary>
        /// Reads the first part of a text file. The header consists of a 
        /// single line with one or two fields: 
        /// [number_of_vectors] number_of_dimensions
        /// The word2vec text format uses both fields.
        /// </summary>
        /// <param name="vocabularyReader">A reader to an open text file</param>
        private void ReadFileHeader(StreamReader vocabularyReader)
        {
            string[] headerFields = ReadLineFields(vocabularyReader);
            int numberOfFields = headerFields.Length;
            switch (numberOfFields)
            {
                case 1:
                    NumberOfDimensions = Int32.Parse(headerFields[0]);
                    break;
                case 2:
                    NumberOfDimensions = Int32.Parse(headerFields[1]);
                    break;
                default:
                    throw new IOException("Vocabulary file format error");
            }
        }

        /// <summary>
        /// Reads labeled vectors from a text file.
        /// </summary>
        /// <param name="vocabularyReader">A reader to an open text file</param>
        private void ReadLabeledVectors(StreamReader vocabularyReader)
        {
            while (!vocabularyReader.EndOfStream)
            {
                string[] entryFields = ReadLineFields(vocabularyReader);
                if (entryFields.Length != NumberOfDimensions + 1)
                {
                    throw new IOException("Vocabulary file format error");
                }

                string label = entryFields[0];
                float[] vector = new float[NumberOfDimensions];
                for (int elementIndex = 0; elementIndex < NumberOfDimensions; elementIndex++)
                {
                    vector[elementIndex] = Single.Parse(entryFields[elementIndex + 1]);
                }
                Add(label, vector);
            }
        }

        /// <summary>
        /// Reads a line and splits it on whitespace into fields.
        /// </summary>
        /// <param name="vocabularyReader">The file stream reader</param>
        /// <returns>The fields of the input line</returns>
        private string[] ReadLineFields(StreamReader vocabularyReader)
        {
            string entry = vocabularyReader.ReadLine();
            return entry.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
        }


    }
}
