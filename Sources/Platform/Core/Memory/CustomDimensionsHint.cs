using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Core.Memory
{
    using TensorDimsOrError = Tuple<TensorDimensions, Exception>;

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
        {
        }

        public CustomDimensionsHint(params int[] dimensions)
            : base(ProcessDimensions(dimensions))
        {
        }

        public CustomDimensionsHint(IEnumerable<int> dimensions)
            : base(ProcessDimensions(dimensions))
        {
        }

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
            return TryComputeDimensions(originalDims.ElementCount) ?? originalDims;
        }

        /// <summary>
        /// In case of error, returns the originalDimensions and a non-empty error message in the output argument.
        /// </summary>
        public TensorDimensions TryToApply(TensorDimensions originalDimensions, out string errorMessage)
        {
            var dimsOrError = ComputeDimensions(originalDimensions.ElementCount);

            errorMessage = dimsOrError.Item2?.Message ?? string.Empty;

            return dimsOrError.Item1 ?? originalDimensions;
        }

        public bool TryToApply(TensorDimensions originalDims, out TensorDimensions customDims)
        {
            TensorDimensions adjustedDims = TryComputeDimensions(originalDims.ElementCount);

            customDims = adjustedDims ?? originalDims;

            return (adjustedDims != null);
        }

        public TensorDimensions Apply(TensorDimensions originalDims)
        {
            var dimsOrError = ComputeDimensions(originalDims.ElementCount);

            if (dimsOrError.Item1 == null)
                throw dimsOrError.Item2;

            return dimsOrError.Item1;
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
                        result = -1; // computed dimension
                    }
                    else if (!int.TryParse(item, out result))
                    {
                        throw new InvalidDimensionsException($"Dimension '{item}' is not an integer.");
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
                    throw new InvalidDimensionsException($"Number {item} is not a valid dimension.");

                if (item == -1)
                {
                    if (foundComputedDimension)
                        throw new InvalidDimensionsException(
                            $"Multiple computed dimensions not allowed (item #{newDimensions.Count + 1}).");

                    foundComputedDimension = true;
                }

                newDimensions.Add(item);

                if (newDimensions.Count > MaxDimensions)
                    throw new InvalidDimensionsException($"Maximum number of dimensions is {MaxDimensions}.");
            }

            return newDimensions.ToImmutable();
        }

        private TensorDimensions TryComputeDimensions(int targetElementCount)
        {
            return ComputeDimensions(targetElementCount).Item1;
        }

        // NOTE: Return some MaybeOrError<T> would be nicer, but Tuples are in the language.
        private TensorDimsOrError ComputeDimensions(int originalElementCount)
        {
            if (IsEmpty || (ElementCount == 0))  // Prevent division by zero.
            {
                return new TensorDimsOrError( null, new InvalidDimensionsException("Custom dimenstions are empty"));
            }

            if (IsFullyDefined)
            {
                // Use the hint only when its element count matches the target count.
                if (ElementCount != originalElementCount)
                {
                    return new TensorDimsOrError(null, new InvalidDimensionsException(
                        $"Original element count ({originalElementCount}) != custom element count ({ElementCount})"));
                }

                // TODO: this superfluously creates a new immutable collection
                return new TensorDimsOrError(new TensorDimensions(m_dims), null); 
            }

            // ...else is not fully defined (there's a computed dimension).
            // Use the hint when target count is divisible by the hint's element count.
            int computed = originalElementCount / ElementCount;

            if (computed*ElementCount != originalElementCount)
            {
                return new TensorDimsOrError(null, new InvalidDimensionsException(
                    $"Original element count {originalElementCount} not divisible by custom element count ({ElementCount})"));
            }

            return new TensorDimsOrError(
                new TensorDimensions(m_dims.Select(dim => (dim == -1) ? computed : dim)), null);
        }

        #endregion
    }
}
