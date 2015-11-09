using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    public class AudioSample
    {
        [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
        public string Audio { get; set; }
        
        [YAXSerializeAs("FileName")]
        [YAXAttributeFor("Audio")]
        [YAXErrorIfMissed(YAXExceptionTypes.Ignore, DefaultValue = "")]
        public string AudioFileName { get; set; }
        
        [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
        public string Image { get; set; }

        [YAXSerializeAs("FileName")]
        [YAXAttributeFor("Image")]
        [YAXErrorIfMissed(YAXExceptionTypes.Ignore, DefaultValue = "")]
        public string ImageFileName { get; set; }
        
        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static AudioSample GetSampleInstance()
        {
            return new AudioSample()
            {
                Audio = "base64",
                AudioFileName = "filesname.jpg",
                Image = "base64",
                ImageFileName = "filesname.jpg"
            };
        }
    }
}
