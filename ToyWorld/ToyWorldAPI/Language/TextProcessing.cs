using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.ToyWorld.Language
{
    /// <summary>
    /// Handles text-based language processing in the Toy World language interface.
    /// Currently, this is limited to simple tokenization.
    /// </summary>
    public class TextProcessing
    {
        public static bool IsEmpty(string text)
        {
            return text == null || text.Length == 0;
        }

        // Tokenizes the input. Currently, this is done by just splitting on white space, 
        // so "@Johnny: What???" returns just two tokens. This is what's needed for AL1, 
        // but future languages may need more sophisticated tokenization.
        public static List<string> Tokenize(string text)
        {
            return text.Split((char[])null, StringSplitOptions.RemoveEmptyEntries).ToList();
        }

        // Tokenizes and returns at most the given number of tokens.
        public static List<string> Tokenize(string text, int maxNumberOfTokens)
        {
            return Tokenize(text).Take(maxNumberOfTokens).ToList();
        }
    }
}
