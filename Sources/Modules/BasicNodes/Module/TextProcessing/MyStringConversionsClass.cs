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

    }
}
