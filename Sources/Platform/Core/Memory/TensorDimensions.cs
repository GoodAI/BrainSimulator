using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Utils;

namespace GoodAI.Core.Memory
{
    public struct TensorDimensions
    {
        private readonly List<int> m_dims;

        private const int MaxDimensions = 100;  // ought to be enough for everybody

        public TensorDimensions(params int[] dimensions)
        {
            m_dims = ProcessDimensions(dimensions);
        }

        public override bool Equals(object obj)
        {
            if (!(obj is TensorDimensions))  // Can't compare value type with null.
                return false;

            return Equals((TensorDimensions)obj);
        }

        public bool Equals(TensorDimensions other)
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

            return m_dims.Aggregate(19, (hashCode, item) => 31*hashCode + item);
        }

        public int Rank
        {
            get
            {
                return (m_dims != null) ? m_dims.Count : 1;
            }
        }

        public int ElementCount
        {
            get
            {
                if (m_dims == null || m_dims.Count == 0)
                    return 0;

                return m_dims.Aggregate(1, (acc, item) => acc*item);
            }
        }

        public int this[int index]
        {
            get
            {
                if ((m_dims == null) && (index == 0))
                    return 0;  // we pretend we have one dimension of size 0

                if (index >= m_dims.Count)
                    throw new IndexOutOfRangeException(string.Format(
                        "Index {0} is greater than max index {1}.", index, m_dims.Count - 1));

                return m_dims[index];
            }
        }

        public string Print(bool printTotalSize = false)
        {
            if (m_dims == null || m_dims.Count == 0)
                return "0";

            return string.Join("×", m_dims.Select(item => item.ToString()))
                + (printTotalSize ? string.Format(" [{0}]", ElementCount) : "");
        }

        public TensorDimensions Transpose()
        {
            if (Rank <= 1)
                return this;

            var transposed = new int[Rank];

            transposed[0] = m_dims[1];
            transposed[1] = m_dims[0];

            for (int i = 2; i < Rank; i++)
                transposed[i] = m_dims[i];

            return new TensorDimensions(transposed);
        }

        public static TensorDimensions GetBackwardCompatibleDims(int count, int columnHint)
        {
            if (count == 0)
                return new TensorDimensions();

            if (columnHint == 0)
                return new TensorDimensions(count);

            // ReSharper disable once InvertIf
            if (count/columnHint*columnHint != count)
            {
                MyLog.WARNING.WriteLine("Count {0} is not divisible by ColumnHint {1}, the hint will be ignored.",
                    count, columnHint);

                return new TensorDimensions(count);
            }

            return new TensorDimensions(columnHint, count/columnHint);
        }

        private static List<int> ProcessDimensions(IEnumerable<int> dimensions)
        {
            var newDimensions = new List<int>();

            foreach (int item in dimensions)
            {
                if ((item <= 0))
                    throw new InvalidDimensionsException(string.Format("Number {0} is not a valid dimension.", item));

                newDimensions.Add(item);

                if (newDimensions.Count > MaxDimensions)
                    throw new InvalidDimensionsException(string.Format("Maximum number of dimensions is {0}.", MaxDimensions));
            }

            return newDimensions;
        }
    }
}
