using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Language;
using Xunit;

namespace ToyWorldTests.Language
{
    public class LanguageOutputTests
    {
        // Size of word vectors
        private const int NumberOfWordVectorDimensions = 50;

        // Initialization
        public LanguageOutputTests()
        {
            Vocabulary.Instance.Initialize(NumberOfWordVectorDimensions);
            
            // Ensure there are some words in the vocabulary
            Vocabulary.Instance.Add("hello");
            Vocabulary.Instance.Add("beautiful");
            Vocabulary.Instance.Add("world"); 
        }

        // Find a vocabulary word given its exact vector
        [Fact]
        public void FindWordFromExactVector()
        {
            const string word = "hello";
            float[] vector = Vocabulary.Instance.VectorFromLabel(word);

            // Recover the vector and its word
            LabeledVector returnVector
                = Vocabulary.Instance.FindNearestNeighbors(vector, 1)[0].Item2;
            Assert.True(vector.SequenceEqual(returnVector.Vector));
            Assert.Equal(returnVector.Label, word);
        }

        // Find a vocabulary word from nearby vector
        [Fact]
        public void FindWordFromApproximateVector()
        {
            const string word = "beautiful";
            float[] vector = Vocabulary.Instance.VectorFromLabel(word);

            // Add a small vector
            float[] nearbyVector = new float[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                nearbyVector[i] = vector[i] + Single.Epsilon;
            }

            // Find the vector and its word from the nearby vector
            LabeledVector returnVector
                = Vocabulary.Instance.FindNearestNeighbors(nearbyVector, 1)[0].Item2;
            Assert.True(vector.SequenceEqual(returnVector.Vector));
            Assert.Equal(returnVector.Label, word);
        }
    }
}
