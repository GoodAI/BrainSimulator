using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication]

    [YAXComment(@"This example shows a multi-level class, which helps to test 
      the null references identity problem. 
      Thanks go to Anton Levshunov for proposing this example,
      and a disussion on this matter.")]
    public class MultilevelClass
    {
        public List<FirstLevelClass> items { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static MultilevelClass GetSampleInstance()
        {
            MultilevelClass obj = new MultilevelClass();
            obj.items = new List<FirstLevelClass>();
            obj.items.Add(new FirstLevelClass());
            obj.items.Add(new FirstLevelClass());

            obj.items[0].Second = new SecondLevelClass();
            obj.items[0].ID = "1";
            obj.items[0].Second.SecondID = "1-2";

            obj.items[1].ID = "2";
            obj.items[1].Second = null;
            return obj;
        }
    }

    public class FirstLevelClass
    {
        public String ID { get; set; }

        public SecondLevelClass Second { get; set; }
    }

    public class SecondLevelClass
    {
        public String SecondID { get; set; }
    }
}
