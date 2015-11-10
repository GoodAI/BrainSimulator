using NUnit.Framework;
using YAXLib;
using YAXLibTests.SampleClasses.PolymorphicSerialization;

namespace YAXLibTests
{
    [TestFixture]
    public class PolymorphicSerializationTests
    {
        [Test]
        [ExpectedException(typeof(YAXPolymorphicException))]
        public void MultipleYaxTypeAttributesWithSameTypeMustThrowAnException()
        {
            var ser = new YAXSerializer(typeof(MultipleYaxTypeAttributesWithSameType));
            var obj = new MultipleYaxTypeAttributesWithSameType();
            ser.Serialize(obj);
        }

        [Test]
        [ExpectedException(typeof(YAXPolymorphicException))]
        public void MultipleYaxTypeAttributesWIthSameAliasMustThrowAnException()
        {
            var ser = new YAXSerializer(typeof(MultipleYaxTypeAttributesWithSameAlias));
            var obj = new MultipleYaxTypeAttributesWithSameAlias();
            ser.Serialize(obj);
        }
    }
}
