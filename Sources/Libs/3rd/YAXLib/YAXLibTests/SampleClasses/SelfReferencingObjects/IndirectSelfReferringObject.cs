using System.Text;

namespace YAXLibTests.SampleClasses.SelfReferencingObjects
{
    public class IndirectSelfReferringObject
    {
        public string ParentDescription { get; set; }
        public ChildReferrenceType Child { get; set; }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine(ParentDescription);
            sb.Append("|->   ").Append(Child).AppendLine();
            return sb.ToString();
        }

        public static IndirectSelfReferringObject GetSampleInstance()
        {
            var parent = new IndirectSelfReferringObject
            {
                ParentDescription = "I'm Parent",
            };
            
            
            var child = new ChildReferrenceType
            {
                ChildDescription = "I'm Child",
            };

            parent.Child = child;
            //child.Parent = parent;
            return parent;
        }

        public static IndirectSelfReferringObject GetSampleInstanceWithLoop()
        {
            var instance = GetSampleInstance();
            instance.Child.Parent = instance;
            return instance;
        }
    }

    public class ChildReferrenceType
    {
        public string ChildDescription { get; set; }
        public IndirectSelfReferringObject Parent { get; set; }

        public override string ToString()
        {
            return ChildDescription;
        }
    }
}
