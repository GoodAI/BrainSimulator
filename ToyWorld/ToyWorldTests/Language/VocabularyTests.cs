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
        // Size of word vectors
        private const int NumberOfWordVectorDimensions = 200;

        // Initialization
        public VocabularyTests()
        {
            Vocabulary.Instance.Initialize(NumberOfWordVectorDimensions);   
        }

        // Loads a word2vec vocabulary
        [Fact]
        public void LoadWord2Vec()
        {
            const int sizeOfVocabulary = 5;

            // Load small vocabulary
            var assembly = Assembly.GetExecutingAssembly();
            var resourceName = "ToyWorldTests.TestFiles.word-vectors.txt";
            using (Stream stream = assembly.GetManifestResourceStream(resourceName))
            using (StreamReader reader = new StreamReader(stream))
            {
                Vocabulary.Instance.Read(reader);
            }

            // Verify vocabulary size
            Assert.Equal(Vocabulary.Instance.Size, sizeOfVocabulary);
        }
    }
}
