using YAXLib;

namespace YAXLibTests.SampleClasses
{
    public class NamesExample
    {
        public string FirstName{get;set;}

        public PersonInfo[] Persons { get; set; }

        public static NamesExample GetSampleInstance()
        {
            PersonInfo info1 = new PersonInfo() { FirstName = "Li" };
            PersonInfo info2 = new PersonInfo() { FirstName = "Hu", LastName = "Hu" };
            NamesExample w = new NamesExample()
            {
                FirstName = "Li",
                Persons = new PersonInfo[] { info1, info2 }
            };

            return w;
        }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }
    }

    public class PersonInfo
    {
        public string FirstName{get;set;}
        public string LastName{get;set;}
    }

}
