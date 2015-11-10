using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Collections;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication]

    [YAXComment("This sample demonstrates serialization of non-generic collection classes")]
    public class NonGenericCollectionsSample
    {
        public List<object> ObjList { get; set; }

        public ArrayList TheArrayList { get; set; }

        public Hashtable TheHashtable { get; set; }

        public Queue TheQueue { get; set; }

        public Stack TheStack { get; set; }

        public SortedList TheSortedList { get; set; }

        public BitArray TheBitArray { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static NonGenericCollectionsSample GetSampleInstance()
        {
            List<object> lst = new List<object>();
            lst.Add(1);
            lst.Add(3.0);
            lst.Add("Hello");
            lst.Add(new DateTime(2010, 3, 4));
            lst.Add(new Author() { Name = "Charles", Age = 50 });

            ArrayList arLst = new ArrayList();
            arLst.Add(2);
            arLst.Add(8.5);
            arLst.Add("Hi");
            arLst.Add(new Author() { Name = "Steve", Age = 30 });

            Hashtable table = new Hashtable();
            table.Add(1.0, "Tim");
            table.Add("Tom", "Sam");
            table.Add(new DateTime(2009, 2, 1), 7);

            BitArray bitArray = new BitArray(10);
            bitArray[1] = true;
            bitArray[6] = true;

            Queue queue = new Queue();
            queue.Enqueue(10);
            queue.Enqueue(20);
            queue.Enqueue(30);

            Stack stack = new Stack();
            stack.Push(100);
            stack.Push(200);
            stack.Push(300);


            SortedList sortedList = new SortedList();
            sortedList.Add(1, 2);
            sortedList.Add(5, 7);
            sortedList.Add(8, 2);

            return new NonGenericCollectionsSample()
            {
                ObjList = lst,
                TheArrayList = arLst,
                TheHashtable = table,
                TheBitArray = bitArray,
                TheQueue = queue,
                TheStack = stack,
                TheSortedList = sortedList
            };
        }

    }
}
