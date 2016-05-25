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
    /// Contains tests relevant to language input, i.e., conversion from 
    /// words to vectors.
    /// </summary>
    public class LanguageInputTests
    {
        // Vocabulary
        private readonly Vocabulary _vocabulary = new Vocabulary();

        // Size of word vectors
        private const int NumberOfWordVectorDimensions = 50;

        // Initialization
        public LanguageInputTests()
        {
            _vocabulary.Initialize(NumberOfWordVectorDimensions);
        }

        // Creates a word vector
        [Fact]
        public void CreateVector()
        {
            float[] vector = _vocabulary.VectorFromLabel("hello");
            Assert.False(Vocabulary.IsZero(vector));
            Assert.Equal(vector.Length, _vocabulary.NumberOfDimensions);
        }

        // Creates different vectors for different words
        [Fact]
        public void CreateDifferentVector()
        {
            float[] vectorHello = _vocabulary.VectorFromLabel("hello");
            float[] vectorWorld = _vocabulary.VectorFromLabel("world");
            Assert.False(vectorWorld.SequenceEqual(vectorHello));
        }

        // Creates identical vectors for identical words
        [Fact]
        public void CreateIdenticalVector()
        {
            float[] vector1 = _vocabulary.VectorFromLabel("hello");
            float[] vector2 = _vocabulary.VectorFromLabel("hello");
            Assert.True(vector1.SequenceEqual(vector2));
        }

    }
}
