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
    /// <description>
    /// Translates vectors to words. Contains a memory block with the nearest 
    /// neighbors of the input vectors. Works only in ToyWorld.
    /// </description>
    public class VectorToWord : MyWorkingNode
    {
        public TranslateVectorToWordTask Translate { get; private set; }
    
        #region Memory Blocks
        /// <summary>
        /// The input word vectors
        /// </summary>
        [MyInputBlock(0)]
        public MyMemoryBlock<float> InputVectors
        {
            get { return GetInput(0); }
        }

        /// <summary>
        /// The output words
        /// </summary>
        [MyOutputBlock(0)]
        public MyMemoryBlock<char> OutputWords
        {
            get { return GetOutput<char>(0); }
            set { SetOutput(0, value); }
        }

        /// <summary>
        /// The labels of the nearest neighbors of each input vector
        /// </summary>
        public MyMemoryBlock<char> NeighborWords { get; private set; }

        /// <summary>
        /// The similarity scores for the nearest neighbors of each input vector
        /// </summary>
        public MyMemoryBlock<float> NeighborSimilarities { get; private set; }

        #endregion

        /// <summary>
        /// The maximum amount of text to output
        /// </summary>
        [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = 128)]
        public int MaxTextLength { get; set; }

        /// <summary>
        /// The maximum length for a word
        /// </summary>
        [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = 20)]
        public int MaxWordLength { get; set; }

        /// <summary>
        /// The maximum number of words to output
        /// </summary>
        [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = 4)]
        public int MaxNumberOfWords { get; set; }

        /// <summary>
        /// The number of neighbors of each word vectors to record in memory block
        /// </summary>
        [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = 10)]
        public int NumberOfNeighbors { get; set; }

        /// <summary>
        /// The word vector vocabulary
        /// </summary>
        internal Vocabulary Vocabulary
        {
            // Will not work if the world is not ToyWorld...
            // Should there be some validation for this?
            get { return (Owner.World as ToyWorld).Vocabulary; }
        }

        /// <summary>
        /// The dimensionality of the word vector space
        /// </summary>
        internal int NumberOfDimensions
        {
            get { return Vocabulary.NumberOfDimensions; }
        }

        /// <summary>
        /// Number of input words
        /// </summary>
        internal int InputWordCount
        {
            get { return InputVectors.Count/NumberOfDimensions; }
        }

        public override void UpdateMemoryBlocks()
        {
            OutputWords.Count = MaxTextLength;
            NeighborSimilarities.Dims = new TensorDimensions(MaxNumberOfWords, NumberOfNeighbors);
            NeighborWords.Dims = new TensorDimensions(MaxNumberOfWords, NumberOfNeighbors, MaxWordLength);
        }

        /// <summary>
        /// Gets the word vector corresponding to the word index.
        /// </summary>
        /// <param name="wordIndex">The index (0..) of the word</param>
        /// <returns>A word vector</returns>
        public float[] GetInputVector(int wordIndex)
        {
            float[] vector = new float[NumberOfDimensions];
            int offset = wordIndex * NumberOfDimensions;
            for (int elementIndex = 0; elementIndex < NumberOfDimensions; elementIndex++)
            {
                vector[elementIndex] = InputVectors.Host[offset + elementIndex];
            }
            return vector;
        }

        /// <summary>
        /// Finds nearest neighbors of a word vector.
        /// </summary>
        /// <param name="vector">Word vector</param>
        /// <returns>List of neighbors and similarities</returns>
        public List<Tuple<float, LabeledVector>> FindNearestNeighbors(float[] vector)
        {
            return Vocabulary.FindNearestNeighbors(vector, NumberOfNeighbors);
        }
    }

    /// <summary>
    /// Translate word vectors into words 
    /// </summary>
    [Description("Translate from vectors to words")]
    public class TranslateVectorToWordTask : MyTask<VectorToWord>
    {
        public override void Init(int nGPU)
        {

        }

        public override void Execute()
        {
            string outputText = "";
            if (Owner.Vocabulary.Size > 0)
            {
                
                Owner.InputVectors.SafeCopyToHost();
                StringBuilder outputTextBuilder = new StringBuilder();
                int maxNumberOfWords = Math.Min(Owner.MaxNumberOfWords, Owner.InputWordCount);
                for (int wordIndex = 0; wordIndex < maxNumberOfWords; wordIndex++)
                {
                    float[] vector = Owner.GetInputVector(wordIndex);
                    var neighbors = Owner.FindNearestNeighbors(vector);
                    //CopyToNeighborsBlocks(wordIndex, neighbors);

                    string nearestNeighborWord = neighbors[0].Item2.Label;
                    AppendWord(outputTextBuilder, nearestNeighborWord);
                }
                outputText = outputTextBuilder.ToString();
            }

            CopyToOutput(outputText);
        }

        private void CopyToNeighborsBlocks(int wordIndex, List<Tuple<float, LabeledVector>> neighbors)
        {
            // transfer neighbors to mem blocks
            // how does it work with observers (we'll need a special observer/UI)
            int neighborIndex = 0;
            foreach (var neighbor in neighbors)
            {
                float similarityMeasure = neighbor.Item1;
                string word = neighbor.Item2.Label;

                Owner.NeighborSimilarities.Host[Flatten(wordIndex, Owner.MaxNumberOfWords, neighborIndex)]
                    = similarityMeasure;

                // copy word

                neighborIndex++;

            }
            // allow for vocabulary size < neighborcount

            Owner.NeighborWords.SafeCopyToDevice();
            Owner.NeighborSimilarities.SafeCopyToDevice();

            throw new NotImplementedException();
        }

        /// <summary>
        /// Turns two dimensions into one array dimension.
        /// </summary>
        /// <param name="index1">First index</param>
        /// <param name="size1">Number of elements along first dimension</param>
        /// <param name="index2">Second index</param>
        /// <returns></returns>
        private int Flatten(int index1, int size1, int index2)
        {
            return index1*size1 + index2;
        }

        /// <summary>
        /// Turns three dimensions into one array dimension.
        /// </summary>
        /// <param name="index1">First index</param>
        /// <param name="size1">Number of elements along first dimension</param>
        /// <param name="index2">Second index</param>
        /// <param name="size2">Number of elements along second dimension</param>
        /// <param name="index3">Third index</param>
        /// <returns></returns>
        private int Flatten(int index1, int size1, int index2, int size2, int index3)
        {
            return index1*size1 + index2*size2 + index3;
        }

        /// <summary>
        /// Copies the text to the output block.
        /// </summary>
        /// <param name="outputText">The text to copy</param>
        private void CopyToOutput(string outputText)
        {
            int charIndex = 0;
            foreach (char ch in outputText)
            {
                Owner.OutputWords.Host[charIndex++] = ch;
            }
            while (charIndex < Owner.OutputWords.Count)
            {
                Owner.OutputWords.Host[charIndex++] = (char)0;
            }
            Owner.OutputWords.SafeCopyToDevice();
        }

        /// <summary>
        /// Appends a word, inserting space between words.
        /// </summary>
        /// <param name="builder">The string builder</param>
        /// <param name="word">The word to add</param>
        private void AppendWord(StringBuilder builder, string word)
        {
            if (builder.Length > 0)
            {
                builder.Append(" ");
            }
            builder.Append(word);
        }
    }
}
