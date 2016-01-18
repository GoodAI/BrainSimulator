using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Core.Memory
{
    public struct TensorDims2
    {
        private List<int> m_dims;

        private const int MaxDimensions = 100;  // ought to be enough for everybody

        public TensorDims2(params int[] dimensions)
        {
            m_dims = ProcessDimensions(dimensions);
        }

        public int Rank
        {
            get { return m_dims.Count; }
        }

        private int ElementCount   // TODO: a better name (Size suggest in bytes ?)
        {
            get
            {
                if (m_dims == null || m_dims.Count == 0)
                    return 0;

                // ReSharper disable once SuggestVarOrType_BuiltInTypes
                int product = 1;
                m_dims.ForEach(item =>
                {
                    if (item != -1)
                        product *= item;
                });

                return product;
            }
        }
        
        public string Print(bool printTotalSize = false)
        {
            if (m_dims == null || m_dims.Count == 0)
                return "0";

            return string.Join("×", m_dims.Select(item => item.ToString()))
                + (printTotalSize ? string.Format(" [{0}]", ElementCount) : "");
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
