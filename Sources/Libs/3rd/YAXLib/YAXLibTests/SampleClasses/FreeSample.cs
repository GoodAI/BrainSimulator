using System.Collections.Generic;

namespace YAXLibTests.SampleClasses
{
    public class FreeSample
    {
        public int BoundViewID { get; set; }
        public decimal SomeDecimalNumber { get; set; }

        public static FreeSample GetSampleInstance()
        {

            return new FreeSample
                       {
                           BoundViewID = 17,
                           SomeDecimalNumber = 12948923849238402394
                       };
        }


        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

    }
}
