using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Core
{
    public class MyConnection
    {
        public int FromIndex { get; private set; }
        public int ToIndex { get; private set; }

        public MyNode From { get; private set; }
        public MyNode To { get; private set; }

        public String Name { get { return From.Name + "_" + To.Name; } }

        public MyConnection(MyNode from, MyNode to, int fromIndex, int toIndex)
        {
            this.From = from;
            this.To = to;
            FromIndex = fromIndex;
            ToIndex = toIndex;
        }

        public MyConnection(MyNode from, MyNode to) : this(from, to, 0, 0) { }

        /*
        public void CheckCompatibility()
        {
            if (!To.GetInfo().InputBlocks[ToIndex].PropertyType.IsAssignableFrom(
                From.GetInfo().OutputBlocks[FromIndex].PropertyType))
            {
                throw new InvalidCastException("Incompatible connection points: (" + From.Name + " -> " + To.Name + ")");
            }
        }
        */
          
        public void Connect()
        {
            if (ToIndex < To.InputConnections.Length && FromIndex < From.OutputBranches)
            {
                To.InputConnections[ToIndex] = this;
            }
            else
            {
                throw new ArgumentException("Bad connection destination (" + From.Name + " -> " + To.Name + ")");
            }           
        }

        public void Disconnect()
        {
            if (ToIndex < To.InputConnections.Length)
            {
                To.InputConnections[ToIndex] = null;
            }
        }

        public MyMemoryBlock<float> FetchInput()
        {
            if (From != null)
            {
                if (FromIndex < From.OutputBranches)
                {
                    return From.GetOutput(FromIndex);
                }
                else return null;
            }
            else return null;
        }

        public MyMemoryBlock<T> FetchInput<T>() where T : struct
        {
            if (From != null)
            {
                if (FromIndex < From.OutputBranches)
                {
                    return From.GetOutput<T>(FromIndex);
                }
                else return null;
            }
            else return null;
        }

        public MyAbstractMemoryBlock FetchAbstractInput()
        {
            if (From != null)
            {
                if (FromIndex < From.OutputBranches)
                {
                    return From.GetAbstractOutput(FromIndex);
                }
                else return null;
            }
            else return null;
        }
    }
}
