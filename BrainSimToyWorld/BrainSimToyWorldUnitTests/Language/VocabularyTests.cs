using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Language;
using Xunit;

namespace ToyWorldTests.Language
{
    /// <summary>
    /// Contains tests for creating the vocabulary.
    /// </summary>
    public class VocabularyTests
    {
        // Vocabulary
        private readonly Vocabulary _vocabulary;

        // Size of word vectors
        private const int NumberOfWordVectorDimensions = 200;

        // Initialization
        public VocabularyTests()
        {
            _vocabulary = new Vocabulary(NumberOfWordVectorDimensions, Vocabulary.WordVectorType.Learned);
        }

        // Loads a word2vec vocabulary
        [Fact]
        public void LoadWord2Vec()
        {
            const int sizeOfVocabulary = 5;

            // Load small vocabulary
            var assembly = Assembly.GetExecutingAssembly();
            var resourceName = "BrainSimToyWorldUnitTests.TestFiles.word-vectors.txt";
            using (Stream stream = assembly.GetManifestResourceStream(resourceName))
            using (StreamReader reader = new StreamReader(stream))
            {
                _vocabulary.Read(reader);
            }

            // Verify vocabulary size
            Assert.Equal(sizeOfVocabulary, _vocabulary.Size);
        }
    }
}
