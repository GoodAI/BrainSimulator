using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace YAXLibTests
{
    public static class StringExtensionsForTest
    {
        public static string StripTypeAssemblyVersion(this string str)
        {
            const string pattern = @"\,\s+Version\=\d+(\.\d+)*\,\s+Culture=\b\w+\b\,\s+PublicKeyToken\=\b\w+\b";
            return Regex.Replace(str, pattern, String.Empty);
        }
    }
}
