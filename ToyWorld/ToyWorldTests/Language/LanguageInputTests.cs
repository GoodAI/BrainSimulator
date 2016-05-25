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
        // Vocabulary with random vectors
        private readonly Vocabulary _randomVectorVocabulary = new Vocabulary();

        // Vocabulary with one-of-N vectors
        private readonly Vocabulary _oneOfNVectorVocabulary = new Vocabulary();

        // Size of word vectors
        private const int NumberOfWordVectorDimensions = 50;

        // Initialization
        public LanguageInputTests()
        {
            _randomVectorVocabulary.Initialize(NumberOfWordVectorDimensions);
            _oneOfNVectorVocabulary.Initialize(NumberOfWordVectorDimensions, Vocabulary.WordVectorType.OneOfN);
        }

        // Creates a word vector
        [Fact]
        public void CreateRandomVector()
        {
            float[] vector = _randomVectorVocabulary.VectorFromLabel("hello");
            Assert.False(Vocabulary.IsZero(vector));
            Assert.Equal(vector.Length, _randomVectorVocabulary.NumberOfDimensions);
        }

        // Creates different vectors for different words
        [Fact]
        public void CreateDifferentRandomVector()
        {
            float[] vectorHello = _randomVectorVocabulary.VectorFromLabel("hello");
            float[] vectorWorld = _randomVectorVocabulary.VectorFromLabel("world");
            Assert.False(vectorWorld.SequenceEqual(vectorHello));
        }

        // Creates identical vectors for identical words
        [Fact]
        public void CreateIdenticalRandomVector()
        {
            float[] vector1 = _randomVectorVocabulary.VectorFromLabel("hello");
            float[] vector2 = _randomVectorVocabulary.VectorFromLabel("hello");
            Assert.True(vector1.SequenceEqual(vector2));
        }

        // Creates a one-of-N vector
        [Fact]
        public void CreateOneOfNVector()
        {
            float[] vector = _oneOfNVectorVocabulary.VectorFromLabel("hello");
            Assert.Equal(1, vector.Sum());
            Assert.Equal(_oneOfNVectorVocabulary.NumberOfDimensions, vector.Length);
        }


    }
}
