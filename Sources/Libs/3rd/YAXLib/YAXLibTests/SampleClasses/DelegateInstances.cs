using System;

namespace YAXLibTests.SampleClasses
{
    public class DelegateInstances
    {
        public delegate string AwesomeDelegate(int n, double d);

        public DelegateInstances()
        {
            Delegate1 = AwesomeMethod;
        }

        public AwesomeDelegate Delegate1 { get; set; }
        public AwesomeDelegate Delegate2 { get; set; }

        public Func<string> SomeFunc { get; set; }
        public Action SomeAction { get; set; }

        public int SomeNumber { get; set; }
        private string AwesomeMethod(int n, double d)
        {
            return "Hi";
        }

        public static DelegateInstances GetSampleInstance()
        {
            return new DelegateInstances
                   {
                        Delegate2 = (n, d) => String.Format("Hey n:{0}, d:{1}", n, d),
                        SomeFunc = () => "Some",
                        SomeAction = () => Console.WriteLine("I'm doing something"),
                        SomeNumber = 12
                   };
        }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }
    }
}
