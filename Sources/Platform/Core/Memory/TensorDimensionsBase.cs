using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Memory;
using GoodAI.Core.Utils;

namespace GoodAI.Core.Memory
{
    public abstract class TensorDimensionsBase
    {
        private readonly IImmutableList<int> m_dims;

        protected const int MaxDimensions = 100;  // ought to be enough for everybody

        protected TensorDimensionsBase()
        {
            m_dims = null;  // This means default dimensions.
        }

        protected TensorDimensionsBase(IImmutableList<int> immutableDimensions)
        {
            m_dims = immutableDimensions;
        }

        public TensorDimensionsBase(params int[] dimensions)
        {
            m_dims = ProcessDimensions(dimensions);
        }

        protected bool Equals(TensorDimensionsBase other)
        {
            if (other.Rank != Rank)
                return false;

            if ((Rank == 1) && (m_dims == null) && (other.m_dims == null))
                return true;

            return (m_dims != null) && (other.m_dims != null) && m_dims.SequenceEqual(other.m_dims);
        }

        public override int GetHashCode()
        {
            if (m_dims == null || m_dims.Count == 0)
                return 0;

            return m_dims.Aggregate(19, (hashCode, item) => 31 * hashCode + item);
        }

        public int Rank
        {
            get
            {
                return (m_dims != null) ? m_dims.Count : 1;
            }
        }

        /// <summary>
        /// Rank synonym. (This is required by tests because this class has an the indexer. Please use Rank if possible.)
        /// </summary>
        [Obsolete("Preferably use Rank instead")]
        public int Count
        { get { return Rank; } }

        public int ElementCount
        {
            get
            {
                if (m_dims == null || m_dims.Count == 0)
                    return 0;

                return m_dims.Aggregate(1, (acc, item) => acc * item);
            }
        }

        public int this[int index]
        {
            get
            {
                if (m_dims == null)
                {
                    if (index == 0)
                        return 0;  // We pretend we have one dimension of size 0.

                    throw GetIndexOutOfRangeException(index, 0);
                }

                if (index >= m_dims.Count)
                    throw GetIndexOutOfRangeException(index, m_dims.Count - 1);

                return m_dims[index];
            }
        }

        private static IndexOutOfRangeException GetIndexOutOfRangeException(int index, int maxIndex)
        {
            return new IndexOutOfRangeException(string.Format(
                "Index {0} is greater than max index {1}.", index, maxIndex));
        }

        public string Print(bool printTotalSize = false)
        {
            if (m_dims == null || m_dims.Count == 0)
                return "0";

            return string.Join("×", m_dims.Select(item => item.ToString()))
                + (printTotalSize ? string.Format(" [{0}]", ElementCount) : "");
        }

        private static IImmutableList<int> ProcessDimensions(IEnumerable<int> dimensions)
        {
            ImmutableList<int>.Builder newDimensionsBuilder = ImmutableList.CreateBuilder<int>();

            foreach (int item in dimensions)
            {
                if (item < 0)
                    throw new InvalidDimensionsException(string.Format("Number {0} is not a valid dimension.", item));

                newDimensionsBuilder.Add(item);

                if (newDimensionsBuilder.Count > MaxDimensions)
                    throw new InvalidDimensionsException(string.Format("Maximum number of dimensions is {0}.", MaxDimensions));
            }

            return newDimensionsBuilder.ToImmutable();
        }
    }
}
