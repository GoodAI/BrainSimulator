using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using System;

namespace GoodAI.Core
{
    public class MyConnection
    {
        public int FromIndex { get; private set; }
        public int ToIndex { get; private set; }

        public MyNode From { get; private set; }
        public MyNode To { get; private set; }

        public String Name { get { return From.Name + "_" + To.Name; } }

        public bool IsLowPriority { get; set; }
        public bool IsHidden { get; set; }

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

        public override bool Equals(object obj)
        {
            var other = obj as MyConnection;
            if (other == null)
                return false;

            return other.From == From
                   && other.To == To
                   && other.FromIndex == FromIndex
                   && other.ToIndex == ToIndex;
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 13;
                hash = (hash*7) + From.GetHashCode();
                hash = (hash*7) + To.GetHashCode();
                hash = (hash*7) + FromIndex.GetHashCode();
                hash = (hash*7) + ToIndex.GetHashCode();

                return hash;
            }
        }

        public void Connect()
        {
            if (ToIndex < To.InputConnections.Length && FromIndex < From.OutputBranches)
            {
                // Check for this before adding the connection to the inputs.
                if (To.InputConnections[ToIndex] != this)
                    // If the flag is already set, keep it, otherwise only set it if the edge would lead to a new cycle.
                    IsLowPriority = IsLowPriority || From.CheckForCycle(To);

                To.InputConnections[ToIndex] = this;
                From.OutputConnections[FromIndex].Add(this);
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
                if (FromIndex < From.OutputConnections.Length)
                    From.OutputConnections[FromIndex].Remove(this);
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
