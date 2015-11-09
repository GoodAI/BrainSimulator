using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication]

    [YAXComment("This example is used in the article to show YAXLib exception handling policies")]
    public class ProgrammingLanguage
    {
        [YAXErrorIfMissed(YAXExceptionTypes.Error)]
        public string LanguageName { get; set; }

        [YAXErrorIfMissed(YAXExceptionTypes.Warning, DefaultValue = true)]
        public bool IsCaseSensitive { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static ProgrammingLanguage GetSampleInstance()
        {
            return new ProgrammingLanguage()
            {
                LanguageName = "C#",
                IsCaseSensitive = true
            };
        }
    }
}
