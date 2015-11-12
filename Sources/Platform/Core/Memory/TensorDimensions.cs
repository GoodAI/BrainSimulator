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
            IsDefault = true;
        }

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

        public void Set(List<int> customDimenstions)
        {
            // TODO: validate, ...
            m_customDimensions = customDimenstions;

            IsDefault = false;
        }

        public override string ToString()  // TODO: remove
        {
            return string.Join(", ",  m_customDimensions);
        }
    }
}
