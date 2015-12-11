using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Core.Memory
{
    public class InvalidDimensionsException : FormatException
    {
        public InvalidDimensionsException(string message) : base(message) { }
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class TensorDimensions : MemBlockAttribute
    {
        [YAXSerializableField, YAXSerializeAs("CustomDimensions")]
        [YAXCollection(YAXCollectionSerializationTypes.Recursive, EachElementName = "Dim")]
        private List<int> m_customDimensions = new List<int>();

        private int m_computedDimension = -1;

        private const string ComputedDimLiteral = "*";

        public TensorDimensions()
        {
            IsCustom = false;
            CanBeComputed = false;
        }

        public TensorDimensions(params int[] dimensions)
        {
            Set(dimensions);

            IsCustom = false;  // do not save dimensions constructed in code
        }

        private const int MaxDimensions = 100; // ought to be enough for everybody

        public int Size
        {
            get { return m_size; }
            set
            {
                m_size = value;
                UpdateComputedDimension();
            }
        }
        private int m_size;

        public int this[int index]
        {
            get
            {
                if (m_customDimensions.Count == 0)
                    return m_size;

                if (index >= m_customDimensions.Count)
                    throw new IndexOutOfRangeException(string.Format(
                        "Index {0} is greater than max index {1}.", index, m_customDimensions.Count - 1));

                return (m_customDimensions[index] != -1) ? m_customDimensions[index] : m_computedDimension;
            }
        }
        
        public int Count
        {
            get { return Math.Max(m_customDimensions.Count, 1); }  // we always have at least one dimension
        }

        public bool CanBeComputed { get; private set; }

        public string LastSetWarning { get; private set; }

        internal override void ApplyAttribute(MyAbstractMemoryBlock memoryBlock)
        {
            memoryBlock.Dims = this;
        }

        public override string ToString()
        {
            return PrintSource();
        }

        public string PrintSource()
        {
            return string.Join(", ", m_customDimensions.Select(item =>
                (item == -1) ? ComputedDimLiteral : item.ToString()
                ));
        }

        public string Print(bool printTotalSize = false, bool indicateComputedDim = false)
        {
            if (m_customDimensions.Count == 0)
                return m_size.ToString();

            string result = string.Join("×", m_customDimensions.Select(item =>
                (item == -1)
                ? string.Format((indicateComputedDim ? "({0})" : "{0}"), PrintComputedDim())
                : item.ToString()
            ));

            return result + (printTotalSize ? string.Format(" [{0}]", Size) : "");
        }

        private string PrintComputedDim()
        {
            return (m_computedDimension == -1) ? "?" : m_computedDimension.ToString();
        }

        public void Set(IEnumerable<int> customDimensions,
            bool autoAddComputedDim = false,
            bool autoRemoveDimsOfSizeOne = false)
        {
            InnerSet(customDimensions, autoAddComputedDim, autoRemoveDimsOfSizeOne);
            
            IsCustom = (m_customDimensions.Count > 0);  // No need to save "empty" value.
        }
        
        /// <summary>
        /// Sets new value but treats it as default (that is not saved to the project). Use for backward compatibility.
        /// </summary>
        public void SetDefault(IEnumerable<int> dimensions, bool autoAddComputedDim = false)
        {
            if (IsCustom)
                return;

            InnerSet(dimensions, autoAddComputedDim, autoRemoveDimsOfSizeOne: true);

            IsCustom = false;  // treat new value as default
        }

        public void Parse(string text)
        {
            if (text.Trim() == string.Empty)
            {
                Set(new List<int>());
                return;
            }

            string[] textItems = text.Split(',', ';');

            IEnumerable<int> dimensions = textItems.Select(item =>
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

            Set(dimensions, autoAddComputedDim: false, autoRemoveDimsOfSizeOne: true);
        }

        private void InnerSet(IEnumerable<int> dimensions, bool autoAddComputedDim, bool autoRemoveDimsOfSizeOne)
        {
            string warning;

            m_customDimensions = ProcessDimensions(dimensions, autoAddComputedDim, out warning);

            LastSetWarning = warning;

            UpdateComputedDimension();
        }

        private static List<int> ProcessDimensions(IEnumerable<int> dimensions, bool autoAddComputedDim, out string warning)
        {
            warning = "";

            var newDimensions = new List<int>();

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
                    throw new InvalidDimensionsException(string.Format("Maximum number of dimensions is {0}.", MaxDimensions));
            }

            // got only the computed dimension, it is equivalent to empty setup
            if (foundComputedDimension && (newDimensions.Count == 1))
            {
                if (string.IsNullOrEmpty(warning))
                    warning = "Only computed dim. changed to empty dimensions.";

                return new List<int>();
            }

            // UX: when no computed dimension was given, let it be the first one
            if (autoAddComputedDim && (newDimensions.Count > 0) && !foundComputedDimension)
                newDimensions.Insert(0, -1);

            return newDimensions;
        }

        private void UpdateComputedDimension()
        {
            m_computedDimension = ComputeDimension();

            CanBeComputed = (m_computedDimension != -1);
        }

        private int ComputeDimension()
        {
            if (m_size == 0)
                return -1;  // don't return dimension size 0

            if (m_customDimensions.Count == 0)
                return m_size;

            int product = 1;
            m_customDimensions.ForEach(item =>
                {
                    if (item != -1)
                        product *= item;
                });

            if (product < 1)
                return -1;

            int computedDimension = m_size / product;

            if (computedDimension * product != m_size)  // unable to compute integer division
                return -1;

            return computedDimension;
        }
    }
}
