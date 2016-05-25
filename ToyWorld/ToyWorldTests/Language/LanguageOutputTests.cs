using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Language;
using Xunit;

namespace ToyWorldTests.Language
{
    /// <summary>
    /// Contains tests relevant to language output, i.e., conversion from 
    /// vectors to words.
    /// </summary>
    public class LanguageOutputTests
    {
        // Vocabulary
        private readonly Vocabulary _vocabulary = new Vocabulary();

        // Size of word vectors
        private const int NumberOfWordVectorDimensions = 50;

        // Initialization
        public LanguageOutputTests()
        {
            _vocabulary.Initialize(NumberOfWordVectorDimensions);
            
            // Ensure there are some words in the vocabulary
            _vocabulary.Add("hello");
            _vocabulary.Add("beautiful");
            _vocabulary.Add("world"); 
        }

        // Find a vocabulary word given its exact vector
        [Fact]
        public void FindWordFromExactVector()
        {
            const string word = "hello";
            float[] vector = _vocabulary.VectorFromLabel(word);

            // Recover the vector and its word
            LabeledVector returnVector
                = _vocabulary.FindNearestNeighbors(vector, 1)[0].Item2;
            Assert.True(vector.SequenceEqual(returnVector.Vector));
            Assert.Equal(returnVector.Label, word);
        }

        // Find a vocabulary word from nearby vector
        [Fact]
        public void FindWordFromApproximateVector()
        {
            const string word = "beautiful";
            float[] vector = _vocabulary.VectorFromLabel(word);

            // Add a small vector
            float[] nearbyVector = new float[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                nearbyVector[i] = vector[i] + Single.Epsilon;
            }

            // Find the vector and its word from the nearby vector
            LabeledVector returnVector
                = _vocabulary.FindNearestNeighbors(nearbyVector, 1)[0].Item2;
            Assert.True(vector.SequenceEqual(returnVector.Vector));
            Assert.Equal(returnVector.Label, word);
        }


        // Find nearest neighbors
        [Fact]
        public void FindNNeighbors()
        {
            const int numberOfNeighbors = 5;

            const string word = "beautiful";
            float[] queryVector = _vocabulary.VectorFromLabel(word);

            // Make increasingly remote neighbors
            float[][] nearbyVectors = new float[numberOfNeighbors][];
            int iVector = 0;
            for (iVector = 0; iVector < numberOfNeighbors; iVector++)
            {
                nearbyVectors[iVector] = new float[NumberOfWordVectorDimensions];
                for (int i = 0; i < queryVector.Length; i++)
                {
                    nearbyVectors[iVector][i] = queryVector[i] + Single.Epsilon * iVector;
                }
                _vocabulary.Add(iVector.ToString(), nearbyVectors[iVector]);
            }
            
            // Get nearest neighbors
            var retrievedNeighbors = _vocabulary.FindNearestNeighbors(queryVector, numberOfNeighbors);

            // Compare to expected neighbors
            iVector = 0;
            foreach (var tuple in retrievedNeighbors)
            {
                Assert.True(tuple.Item2.Vector.SequenceEqual(nearbyVectors[iVector++]));
            }
        }

    }
}
