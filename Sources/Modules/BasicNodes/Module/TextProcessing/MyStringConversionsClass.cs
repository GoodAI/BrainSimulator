using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Modules.LTM
{
    public static class MyStringConversionsClass
    {
        /// <summary>
        /// Enumeration of different encoding schemes that can be used by text processing nodes
        /// </summary>
        public enum StringEncodings { DigitIndexes, UVSC }

        /// <summary>
        /// Different ways to make sure that a word takes up all the space it can
        /// </summary>
        public enum PaddingSchemes { None, Repeat, Stretch }

        /// <summary>
        /// Converts char to index in ASCII table. Inverse function to DigitIndexToString().
        /// </summary>
        /// <param name="str">Input char.</param>
        /// <returns>Index of char in ASCII table minus constant 32.</returns>
        public static int StringToDigitIndexes(char str)
        {
            int res = 0;
            int charValue = str;
            if (charValue >= ' ' && charValue <= '~')
                res = charValue - ' ';
            else
            {
                if (charValue == '\n')
                    res = '~' - ' ' + 1;
            }
            return res;
        }

        /// <summary>
        /// Converts an index back to char. Inverse function to StringToDigitIndexes().
        /// </summary>
        /// <param name="code"> Index of the character. Should be a whole number. TODO:</param>
        /// <returns></returns>
        public static char DigitIndexToString(float code)
        {
           
            if (code == '~' - ' ' + 1)
                return '\n';

            if (code < 0 || code + ' ' > '~')
            {
                GoodAI.Core.Utils.MyLog.WARNING.WriteLine("Unrecognized code '" + code + "' for conversion to character.");
                return ' ';
            }

            return (char)(' ' + code);
        }

        /// <author>GoodAI</author>
        /// <summary>
        /// Convert input char from ASCII to Uppercase Vowel-Space-Consonant encoding
        /// The coding also takes letter similarity and frequency into account
        /// </summary>
        /// <meta>vkl</meta>
        /// <param name="input">Char to convert</param>
        /// <returns>Encoded character number</returns>
        public static int StringToUvscCoding(char input)
        {
            switch (input)
            {
                case 'u': case 'U':
                    return 0;
                case 'i': case 'I':
                    return 1;
                case 'o': case 'O':
                    return 2;
                case 'a': case 'A':
                    return 3;
                case 'e': case 'E':
                    return 4;
                case ' ':
                    return 5;
                case 't': case 'T':
                    return 6;
                case 'd': case 'D':
                    return 7;
                case 'n': case 'N':
                    return 8;
                case 'm': case 'M':
                    return 9;
                case 'b': case 'B':
                    return 10;
                case 'p': case 'P':
                    return 11;
                case 's': case 'S':
                    return 12;
                case 'r': case 'R':
                    return 13;
                case 'h': case 'H':
                    return 14;
                case 'v': case 'V':
                    return 15;
                case 'w': case 'W':
                    return 16;
                case 'c': case 'C':
                    return 17;
                case 'k': case 'K':
                    return 18;
                case 'g': case 'G':
                    return 19;
                case 'f': case 'F':
                    return 20;
                case 'j': case 'J':
                    return 21;
                case 'l': case 'L':
                    return 22;
                case 'q': case 'Q':
                    return 23;
                case 'z': case 'Z':
                    return 24;
                case 'x': case 'X':
                    return 25;
                case 'y': case 'Y':
                    return 26;

                default:
                    return 5;
            }
        }

        /// <author>GoodAI</author>
        /// <summary>
        /// Convert input char from Uppercase Vowel-Space-Consonant encoding to ASCII string.
        /// Inverse of StringToUvscCoding.
        /// </summary>
        /// <meta>vkl</meta>
        /// <param name="input">Char to convert</param>
        /// <returns>Encoded character number</returns>
        public static char UvscCodingToString(float input)
        {
            if (input < 0.5f)
                return 'U';
            else if (input < 1.5f)
                return 'I';
            else if (input < 2.5f)
                return 'O';
            else if (input < 3.5f)
                return 'A';
            else if (input < 4.5f)
                return 'E';
            else if (input < 5.5f)
                return ' ';
            else if (input < 6.5f)
                return 'T';
            else if (input < 7.5f)
                return 'D';
            else if (input < 8.5f)
                return 'N';
            else if (input < 9.5f)
                return 'M';
            else if (input < 10.5f)
                return 'B';
            else if (input < 11.5f)
                return 'P';
            else if (input < 12.5f)
                return 'S';
            else if (input < 13.5f)
                return 'R';
            else if (input < 14.5f)
                return 'H';
            else if (input < 15.5f)
                return 'V';
            else if (input < 16.5f)
                return 'W';
            else if (input < 17.5f)
                return 'C';
            else if (input < 18.5f)
                return 'K';
            else if (input < 19.5f)
                return 'G';
            else if (input < 20.5f)
                return 'F';
            else if (input < 21.5f)
                return 'J';
            else if (input < 22.5f)
                return 'L';
            else if (input < 23.5f)
                return 'Q';
            else if (input < 24.5f)
                return 'Z';
            else if (input < 25.5f)
                return 'X';
            else if (input < 26.5f)
                return 'Y';
            else
                return ' ';
        }

        /// <author>GoodAI</author>
        /// <summary>
        /// Converts UVSC coding to digit indexes. TODO: Direct conversion would probably be more efficient
        /// </summary>
        /// <meta>vkl</meta>
        /// <param name="input">Character to convert, in UVSC coding</param>
        /// <returns>Character in digit index</returns>
        public static int UvscCodingToDigitIndexes(float input)
        {
            return StringToDigitIndexes(UvscCodingToString(input));
        }

        /// <author>GoodAI</author>
        /// <summary>
        /// Takes an input string and returns space-seperated string that 
        /// repeats the word until TextWidth characters are used
        /// </summary>
        /// <meta>vkl</meta>
        /// <param name="input">String to convert</param>
        /// <returns>Converted string</returns>
        public static string RepeatWord(string input, int textWidth)
        {
            StringBuilder output = new StringBuilder(textWidth);

            int p = 0;
            for (int i = 0; i < textWidth; i++)
            {
                output.Append(input[p]);

                if (input[p] == ' ')
                {
                    p = 0;
                    continue;
                }

                p++;
            }

            return output.ToString();
        }

        /// <author>GoodAI</author>
        /// <summary>
        /// Takes an input string and stretches is by repeating letters until it is long enough to fill textWidth.
        /// </summary>
        /// <meta>vkl</meta>
        /// <param name="input">String to convert</param>
        /// <param name="textWidth">Converted string</param>
        /// <returns></returns>
        public static string StretchWord(string input, int textWidth)
        {
            StringBuilder output = new StringBuilder(textWidth);

            int inputLength = input.IndexOf(' ');
            if (inputLength == -1)
                inputLength = input.Length;

            float ratio = ((float)inputLength) / ((float)textWidth);

            for (int i = 0; i < textWidth; i++)
            {
                int sourceIndex = (int)(ratio * i);
                output.Append(input[sourceIndex]);
            }

            return output.ToString();
        }
    }
}
