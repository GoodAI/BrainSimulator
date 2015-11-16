using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Core.Memory
{
    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class TensorDimensions : MemBlockAttribute
    {
        [YAXSerializableField, YAXSerializeAs("CustomDimensions")]
        [YAXCollection(YAXCollectionSerializationTypes.Recursive, EachElementName = "Dim")]
        private List<int> m_customDimensions = new List<int>();

        private int m_computedDimension = -1;

        public TensorDimensions()
        {
            IsCustom = false;
        }

        public readonly int MaxDimensions = 10;

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

        // TODO: an indexer?
        
        public int Count
        {
            get { return m_customDimensions.Count; }
        }

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
                (item == -1) ? "_" : item.ToString()
                ));
        }

        public string PrintResult()
        {
            return string.Join("×", m_customDimensions.Select(item =>
                {
                    if (item == -1)
                    {
                        return "(" + ((m_computedDimension == -1)
                                ? "?"
                                : m_computedDimension.ToString())
                            + ")";
                    }
                    else
                    {
                        return item.ToString();
                    }
                }));
        }

        public void Set(IEnumerable<int> customDimenstions)
        {
            InnerSet(customDimenstions);
            
            IsCustom = true;
        }

        public void Parse(string text)
        {
            var textItems = text.Split(new char[] {',', ';' });

            var dimensions = textItems.Select(item =>
            {
                int result;

                if (item.Trim() == "_")
                {
                    result = -1;  // computed dimension
                }
                else if (!int.TryParse(item.Trim(), out result))
                {
                    throw new FormatException(string.Format("Dimension '{0}' is not an integer.", item));
                }

                return result;
            });

            Set(dimensions);
        }

        private void InnerSet(IEnumerable<int> dimensions)
        {
            var newDimensions = new List<int>();

            bool foundComputedDimension = false;
            int count = 0;

            foreach (var item in dimensions)
            {
                if ((item < -1) || (item == 0))
                    throw new FormatException(string.Format("Number {0} is not a valid dimension.", item));

                if (item == -1)
                {
                    if (foundComputedDimension)
                        throw new FormatException(
                            string.Format("Multiple computed dimensions not allowed (item #{0}).", count + 1));

                    foundComputedDimension = true;
                }
                
                count++;
                if (count > MaxDimensions)
                    throw new FormatException(string.Format("Maximum number of dimensions is {0}.", MaxDimensions));

                newDimensions.Add(item);
            }

            // UX: when no computed dimension was given, let it be the first one
            if (!foundComputedDimension)
                newDimensions.Insert(0, -1);

            m_customDimensions = newDimensions;

            UpdateComputedDimension();
        }

        private void UpdateComputedDimension()
        {
            m_computedDimension = ComputeDimension(m_size);
        }

        private int ComputeDimension(int size)
        {
            if (size == 0)
                return -1;  // don't return dimension size 0

            if (m_customDimensions.Count == 0)
                return size;

            int product = 1;
            m_customDimensions.ForEach(item =>
                {
                    if (item != -1)
                        product *= item;
                });

            if (product < 1)
                return -1;

            var computedDimension = size / product;

            if (computedDimension * product != size)  // unable to compute integer division
                return -1;

            return computedDimension;
        }
    }
}
