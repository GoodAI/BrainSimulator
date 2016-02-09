using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Core.Memory
{
    public class CustomDimensionsHint : TensorDimensionsBase
    {
        #region Static

        private static CustomDimensionsHint m_emptyInstance;

        public static CustomDimensionsHint Empty
        {
            get { return m_emptyInstance ?? (m_emptyInstance = new CustomDimensionsHint()); }
        }

        #endregion

        public CustomDimensionsHint()
        {}

        public CustomDimensionsHint(IEnumerable<int> dimensions)
            : base(ProcessDimensions(dimensions))
        {}

        public CustomDimensionsHint(string source)
            : base(ProcessDimensions(Parse(source)))
        {}

        public bool IsFullyDefined
        {
            get { return (m_dims != null) && m_dims.All(dim => (dim != -1)); }
        }

        #region Object overrides

        public override bool Equals(object obj)
        {
            if (!(obj is CustomDimensionsHint))
                return false;

            return base.Equals((CustomDimensionsHint)obj);
        }

        public bool Equals(CustomDimensionsHint dimensionsHint)
        {
            return base.Equals(dimensionsHint);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        #endregion

        #region Public

        public TensorDimensions ComputeDimensions(int targetElementCount)
        {
            if (ElementCount == 0)  // I'm empty. (Prevent division by zero.)
            {
                return new TensorDimensions(targetElementCount);
            }

            if (IsFullyDefined && (ElementCount == targetElementCount))
            {
                return new TensorDimensions(m_dims); // TODO: this superfluously creates a new immutable collection
            }

            int computed = targetElementCount / ElementCount;

            return (computed*ElementCount == targetElementCount)
                ? new TensorDimensions(m_dims.Select(dim => (dim == -1) ? computed : dim))
                : new TensorDimensions(targetElementCount);
        }

        #endregion

        #region Private

        private const string ComputedDimLiteral = "*";

        private static IEnumerable<int> Parse(string source)
        {
            if (source.Trim() == string.Empty)
            {
                return Enumerable.Empty<int>();
            }

            string[] sourceItems = source.Split(',', ';');

            IEnumerable<int> dimensions = sourceItems.Select(item =>
            {
                int result;

                if ((item.Trim() == ComputedDimLiteral) || (item.Trim() == "_"))
                {
                    result = -1;  // computed dimension
                }
                else if (!int.TryParse(item.Trim(), out result))
                {
                    throw new InvalidDimensionsException(string.Format("Dimension '{0}' is not an integer.", item));
                }

                return result;
            });

            return dimensions;
        }

        private static ImmutableList<int> ProcessDimensions(IEnumerable<int> dimensions)
        {
            var newDimensions = ImmutableList.CreateBuilder<int>();

            bool foundComputedDimension = false;

            foreach (int item in dimensions)
            {
                if ((item < -1) || (item == 0))
                    throw new InvalidDimensionsException(string.Format("Number {0} is not a valid dimension.", item));

                if (item == -1)
                {
                    if (foundComputedDimension)
                        throw new InvalidDimensionsException(string.Format(
                            "Multiple computed dimensions not allowed (item #{0}).", newDimensions.Count + 1));

                    foundComputedDimension = true;
                }

                newDimensions.Add(item);

                if (newDimensions.Count > MaxDimensions)
                    throw new InvalidDimensionsException(string.Format("Maximum number of dimensions is {0}.",
                        MaxDimensions));
            }

            return newDimensions.ToImmutable();
        }

        #endregion
    }
}
