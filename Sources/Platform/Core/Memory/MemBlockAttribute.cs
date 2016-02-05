using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Core.Memory
{
    // TODO(Premek): Remove class.
    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public abstract class MemBlockAttribute
    {
        /// <summary>
        /// Has a non-default value. Attribute with a default value will not be saved, only custom values will be.
        /// </summary>
        [YAXDontSerialize]
        public bool IsCustom { get; set; }

        internal abstract void ApplyAttribute(MyAbstractMemoryBlock myAbstractMemoryBlock);

        internal virtual void FinalizeDeserialization()
        {
            IsCustom = true;  // make sure to be marked as having non-default value
        }
    }
}
