using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Utils;

namespace GoodAI.Core.Memory
{
    public class TensorDimensions : TensorDimensionsBase
    {
        #region Static 

        private static TensorDimensions m_emptyInstance;

        public static TensorDimensions Empty
        {
            get { return m_emptyInstance ?? (m_emptyInstance = new TensorDimensions()); }
        }

        #endregion

        public TensorDimensions()
        {}

        public TensorDimensions(params int[] dimensions) : base(ProcessDimensions(dimensions))
        {}

        public TensorDimensions(IEnumerable<int> dimensions) : base(ProcessDimensions(dimensions))
        {}

        public override bool Equals(object obj)
        {
            if (!(obj is TensorDimensions))
                return false;

            return base.Equals((TensorDimensions)obj);
        }

        public bool Equals(TensorDimensions dimensionsHint)
        {
            return base.Equals(dimensionsHint);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
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
            if (IsEmpty)
                return TensorDimensions.Empty;

            if (Rank == 1)
                return new TensorDimensions(1, m_dims[0]);  // Row vector -> column vector.

            if (Rank < 2)
                throw new InvalidOperationException(string.Format("Invalid Rank value {0}", Rank));

            var transposed = new int[Rank];

            transposed[0] = m_dims[1];
            transposed[1] = m_dims[0];

            for (var i = 2; i < Rank; i++)
                transposed[i] = m_dims[i];

            return new TensorDimensions(transposed);
        }

        public static TensorDimensions GetBackwardCompatibleDims(int count, int columnHint)
        {
            if (count == 0)
                return Empty;

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
