using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication]

    [YAXComment("This example shows how to choose the fields to be serialized")]
    [YAXSerializableType(FieldsToSerialize=YAXSerializationFields.AttributedFieldsOnly)]
    public class FieldSerializationExample
    {
        [YAXSerializableField]
        private int m_someInt;

        [YAXSerializableField]
        private double m_someDouble;

        [YAXSerializableField]
        private string SomePrivateStringProperty { get; set; }

        public string SomePublicPropertyThatIsNotSerialized { get; set; }

        public FieldSerializationExample()
        {
            m_someInt = 8;
            m_someDouble = 3.14;
            SomePrivateStringProperty = "Hi";
            SomePublicPropertyThatIsNotSerialized = "Public";
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            sb.AppendLine("m_someInt: " + m_someInt);
            sb.AppendLine("m_someDouble: " + m_someDouble);
            sb.AppendLine("SomePrivateStringProperty: " + SomePrivateStringProperty);
            sb.AppendLine("SomePublicPropertyThatIsNotSerialized: " + SomePublicPropertyThatIsNotSerialized);

            return sb.ToString();
        }

        public static FieldSerializationExample GetSampleInstance()
        {
            return new FieldSerializationExample();
        }
    }
}
