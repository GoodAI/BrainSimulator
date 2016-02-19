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

        public static CustomDimensionsHint Parse(string source)
        {
            return new CustomDimensionsHint(ParseToEnumerable(source));
        }

        #endregion

        private CustomDimensionsHint()
        {}

        public CustomDimensionsHint(params int[] dimensions)
            : base(ProcessDimensions(dimensions))
        {}

        public CustomDimensionsHint(IEnumerable<int> dimensions)
            : base(ProcessDimensions(dimensions))
        {}

        public bool IsFullyDefined
        {
            get { return !IsEmpty && m_dims.All(dim => (dim != -1)); }
        }

        #region Object overrides

        public override bool Equals(object obj)
        {
            var customDimensionsHint = obj as CustomDimensionsHint;

            return (customDimensionsHint != null) && base.Equals(customDimensionsHint);
        }

        public bool Equals(CustomDimensionsHint dimensionsHint)
        {
            return base.Equals(dimensionsHint);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        public override string ToString()
        {
            return PrintSource();
        }

        #endregion

        #region Public

        public TensorDimensions TryToApply(TensorDimensions originalDims)
        {
            return ComputeDimensions(originalDims.ElementCount) ?? originalDims;
        }

        public bool TryToApply(TensorDimensions originalDims, out TensorDimensions customDims)
        {
            TensorDimensions adjustedDims = ComputeDimensions(originalDims.ElementCount);

            customDims = adjustedDims ?? originalDims;

            return (adjustedDims != null);
        }

        public string PrintSource()
        {
            if (IsEmpty)
                return "";

            return string.Join(", ", m_dims.Select(item =>
                (item == -1) ? ComputedDimLiteral : item.ToString()
                ));
        }

        #endregion

        #region Private

        private const string ComputedDimLiteral = "*";

        private static IEnumerable<int> ParseToEnumerable(string source)
        {
            if (source.Trim() == string.Empty)
            {
                return Enumerable.Empty<int>();
            }

            IEnumerable<int> dimensions = source.Split(',', ';')
                .Select(item => item.Trim())
                .Select(item =>
            {
                int result;

                if (item == ComputedDimLiteral)
                {
                    result = -1;  // computed dimension
                }
                else if (!int.TryParse(item, out result))
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

        private TensorDimensions ComputeDimensions(int targetElementCount)
        {
            if (IsEmpty || (ElementCount == 0))  // Prevent division by zero.
            {
                return null;
            }

            if (IsFullyDefined)
            {
                // Use the hint only when its element count matches the target count.
                return (ElementCount == targetElementCount)
                    ? new TensorDimensions(m_dims) // TODO: this superfluously creates a new immutable collection
                    : null;
            }

            // ...else is not fully defined (there's a computed dimension).
            // Use the hint when target count is divisible by the hint's element count.
            int computed = targetElementCount / ElementCount;

            return (computed * ElementCount == targetElementCount)
                ? new TensorDimensions(m_dims.Select(dim => (dim == -1) ? computed : dim))
                : null;
        }

        #endregion
    }
}
