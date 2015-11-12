using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Core.Memory
{
    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public abstract class MemBlockAttribute
    {
        /// <summary>
        /// Attribute with a default value will not be saved.
        /// </summary>
        [YAXDontSerialize]
        public bool IsDefault { get; protected set; }

        internal abstract void ApplyAttribute(MyAbstractMemoryBlock myAbstractMemoryBlock);
    }
}
