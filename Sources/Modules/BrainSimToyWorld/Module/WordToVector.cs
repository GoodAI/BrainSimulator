using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.ToyWorld.Language;
using YAXLib;

namespace GoodAI.ToyWorld
{
    /// <author>Simon Andersson</author>
    /// <status>Work in progress</status>
    /// <summary>Word to vector node</summary>
    /// <description>Translates words to vectors. Only works in ToyWorld.</description>
    public class WordToVector : MyWorkingNode
    {
        /// <summary>
        /// Translate words into vectors
        /// </summary>
        public TranslateWordToVectorTask Translate { get; private set; }

        #region Memory Blocks
        /// <summary>
        /// The input text.
        /// While the data are chars, they are currenly stored as floats for 
        /// compatibility with nodes that use floats.
        /// </summary>
        [MyInputBlock(0)]
        public MyMemoryBlock<float> TextInput
        {
            get { return GetInput(0); }
        }

        /// <summary>
        /// The output vectors
        /// </summary>
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }
        #endregion

        /// <summary>
        /// The maximum number of words in the text input buffer
        /// </summary>
        [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = 4)]
        public int MaxNumberOfWords { get; set; }

        /// <summary>
        /// The word vector vocabulary
        /// </summary>
        internal Vocabulary Vocabulary {
            get
            {
                // Will not work if the world is not ToyWorld...
                // Should there be some validation for this?
                return (Owner.World as ToyWorld).Vocabulary;
            } 
        }

        /// <summary>
        /// The dimensionality of the word vector space
        /// </summary>
        internal int NumberOfDimensions
        {
            get
            {
                return Vocabulary.NumberOfDimensions;
            }
        }

        /// <summary>
        /// The input text as an array of words
        /// </summary>
        internal string[] InputWords { get; set; }

        public override void UpdateMemoryBlocks()
        {
            Output.Dims = new TensorDimensions(NumberOfDimensions, MaxNumberOfWords);
        }

        /// <summary>
        /// Update the input words if necessary
        /// </summary>
        internal void UpdateWordInput()
        {
            InputWords = SplitTextInput(GetStringFromInput());
        }

        /// <summary>
        /// Extract a string from the input text 
        /// </summary>
        /// <returns>Input text as a string</returns>
        private string GetStringFromInput()
        {
            TextInput.SafeCopyToHost();

            StringBuilder builder = new StringBuilder();
            for (int charIndex = 0;
                charIndex < TextInput.Count && TextInput.Host[charIndex] != 0;
                charIndex++)
            {
                builder.Append((char)TextInput.Host[charIndex]);
            }
            return builder.ToString();
        }

        /// <summary>
        /// Split the input text into words
        /// </summary>
        /// <param name="input">Input text</param>
        /// <returns>An array of words</returns>
        private string[] SplitTextInput(string input)
        {
            return input.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
        }
    }

    /// <summary>
    /// Translate words into word vectors
    /// </summary>
    [Description("Translate from words to vectors")]
    public class TranslateWordToVectorTask : MyTask<WordToVector>
    {
        public override void Init(int nGPU)
        {

        }

        public override void Execute()
        {
            Owner.UpdateWordInput();

            int outputIndex = 0;
            foreach (string word in Owner.InputWords)
            {
                // Need to test for the case where actual number of words > MaxNumberOfWords
                if (outputIndex < Owner.Output.Count)
                {
                    float[] vector = Owner.Vocabulary.VectorFromLabel(word);
                    foreach (float element in vector)
                    {
                        Owner.Output.Host[outputIndex++] = element;
                    }
                }
            }
            while (outputIndex < Owner.Output.Count)
            {
                Owner.Output.Host[outputIndex++] = 0;
            }

            Owner.Output.SafeCopyToDevice();
        }
    }
}
