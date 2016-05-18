using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Language;
using Xunit;

namespace ToyWorldTests.Language
{
    public class LanguageInputTests
    {
        // Size of word vectors
        private const int NumberOfWordVectorDimensions = 50;

        // Initialization
        public LanguageInputTests()
        {
            Vocabulary.Instance.Initialize(NumberOfWordVectorDimensions);   
        }

        // Creates a word vector
        [Fact]
        public void CreateVector()
        {
            float[] vector = Vocabulary.Instance.VectorFromLabel("hello");
            Assert.False(Vocabulary.IsZero(vector));
            Assert.Equal(vector.Length, Vocabulary.Instance.NumberOfDimensions);
        }

        // Creates different vectors for different words
        [Fact]
        public void CreateDifferentVector()
        {
            float[] vectorHello = Vocabulary.Instance.VectorFromLabel("hello");
            float[] vectorWorld = Vocabulary.Instance.VectorFromLabel("world");
            Assert.False(vectorWorld.SequenceEqual(vectorHello));
        }

        // Creates identical vectors for identical words
        [Fact]
        public void CreateIdenticalVector()
        {
            float[] vector1 = Vocabulary.Instance.VectorFromLabel("hello");
            float[] vector2 = Vocabulary.Instance.VectorFromLabel("hello");
            Assert.True(vector1.SequenceEqual(vector2));
        }
    }
}
