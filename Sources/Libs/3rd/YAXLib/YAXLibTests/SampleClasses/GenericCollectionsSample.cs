using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication]

    [YAXComment(@"This class provides an example of successful serialization/deserialization 
        of collection objects in ""System.Collections.Generic"" namespaces")]
    public class GenericCollectionsSample
    {
        public Stack<int> TheStack { get; set; }

        [YAXDictionary(EachPairName = "Item", SerializeKeyAs = YAXNodeTypes.Attribute, SerializeValueAs = YAXNodeTypes.Attribute)]
        public SortedList<double, string> TheSortedList { get; set; }

        [YAXDictionary(EachPairName = "Item", SerializeKeyAs = YAXNodeTypes.Attribute, SerializeValueAs = YAXNodeTypes.Attribute)]
        public SortedDictionary<int, double> TheSortedDictionary { get; set; }

        public Queue<string> TheQueue { get; set; }
        public HashSet<int> TheHashSet { get; set; }
        public LinkedList<double> TheLinkedList { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static GenericCollectionsSample GetSampleInstance()
        {
            Stack<int> stack = new Stack<int>();
            stack.Push(7);
            stack.Push(1);
            stack.Push(79);

            SortedList<double, string> sortedList = new SortedList<double, string>();
            sortedList.Add(1.0, "Hi");
            sortedList.Add(0.5, "Hello");
            sortedList.Add(5.0, "How are you?");

            SortedDictionary<int, double> sortedDic = new SortedDictionary<int, double>();
            sortedDic.Add(5, 2.0);
            sortedDic.Add(10, 1.0);
            sortedDic.Add(1, 30.0);

            Queue<string> q = new Queue<string>();
            q.Enqueue("Hi");
            q.Enqueue("Hello");
            q.Enqueue("How are you?");

            HashSet<int> hashSet = new HashSet<int>();
            hashSet.Add(1);
            hashSet.Add(2);
            hashSet.Add(4);
            hashSet.Add(6);

            LinkedList<double> lnkList = new LinkedList<double>();
            lnkList.AddLast(1.0);
            lnkList.AddLast(5.0);
            lnkList.AddLast(61.0);

            return new GenericCollectionsSample()
            {
                TheStack = stack,
                TheSortedList = sortedList,
                TheSortedDictionary = sortedDic,
                TheQueue = q,
                TheHashSet = hashSet,
                TheLinkedList = lnkList
            };
        }
    }
}
