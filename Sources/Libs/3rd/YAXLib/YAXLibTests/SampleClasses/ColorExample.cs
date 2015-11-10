using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication]

    [YAXComment("This example shows a technique for serializing classes without a default constructor")] 
    public class ColorExample
    {
        private Color m_color = Color.Blue;

        public string TheColor
        {
            get
            {
                return String.Format("#{0:X}", m_color.ToArgb());
            }

            set
            {
                m_color = Color.White;

                value = value.Trim();
                if (value.StartsWith("#")) // remove leading # if any
                    value = value.Substring(1);

                int n;
                if (Int32.TryParse(value, System.Globalization.NumberStyles.HexNumber, null, out n))
                {
                    m_color = Color.FromArgb(n);
                }
            }
        }

        public override string ToString()
        {
            //return GeneralToStringProvider.GeneralToString(this);
            return String.Format("TheColor: {0}", m_color.ToString());
        }

        public static ColorExample GetSampleInstance()
        {
            return new ColorExample();
        }
    }
}
