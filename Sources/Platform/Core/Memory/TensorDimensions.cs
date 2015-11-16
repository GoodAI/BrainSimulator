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

        public TensorDimensions()
        {
            IsCustom = false;
        }

        public readonly int MaxDimensions = 10;

        public int Size { get; private set; }

        // an indexer
        
        public int Count
        {
            get { return m_customDimensions.Count; }
        }

        internal override void ApplyAttribute(MyAbstractMemoryBlock memoryBlock)
        {
            memoryBlock.Dims = this;
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
        }

        public override string ToString()  // TODO: remove
        {
            return string.Join(", ",  m_customDimensions);
        }
    }
}
