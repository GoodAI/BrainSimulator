using System.Drawing;
using System.Xml.Linq;

using NUnit.Framework;

using YAXLib;
using System;
using YAXLibTests.SampleClasses;

namespace YAXLibTests
{
    [TestFixture]
    public class KnownTypeTests
    {
        [Test]
        public void TestExtensionMethod()
        {
            var colorKnownType = new ColorDynamicKnownType();
            var t1 = colorKnownType.Type;
            IKnownType kt = new ColorDynamicKnownType();

            Assert.That(kt.Type, Is.EqualTo(t1));
        }

        [Test]
        public void TestColorNames()
        {
            var colorKnownType = new ColorDynamicKnownType();

            var elem = new XElement("TheColor", "Red");
            var desCl = (Color)colorKnownType.Deserialize(elem, String.Empty);
            Assert.That(desCl.ToArgb(), Is.EqualTo(Color.Red.ToArgb()));

            var serElem = new XElement("TheColor");
            colorKnownType.Serialize(Color.Red, serElem, "");
            Assert.That(serElem.ToString(), Is.EqualTo(elem.ToString()));

            var elemRgbForRed = new XElement("TheColor",
                new XElement("A", 255),
                new XElement("R", 255),
                new XElement("G", 0),
                new XElement("B", 0));
            var desCl2 = (Color)colorKnownType.Deserialize(elemRgbForRed, "");
            Assert.That(desCl2.ToArgb(), Is.EqualTo(Color.Red.ToArgb()));

            var elemRgbAndValueForRed = new XElement("TheColor",
                "Blue",
                new XElement("R", 255),
                new XElement("G", 0),
                new XElement("B", 0));
            var desCl3 = (Color)colorKnownType.Deserialize(elemRgbAndValueForRed, "");
            Assert.That(desCl3.ToArgb(), Is.EqualTo(Color.Red.ToArgb()));
        }

        [Test]
        public void TestWrappers()
        {
            var typeToTest = typeof (TimeSpan);
            var serializer = new YAXSerializer(typeToTest);
            var typeWrapper = new UdtWrapper(typeToTest, serializer);

            Assert.That(typeWrapper.IsKnownType, Is.True);
        }

        [Test]
        public void TestSingleKnownTypeSerialization()
        {
            var typeToTest = typeof(Color);
            var serializer = new YAXSerializer(typeToTest);

            var col1 = Color.FromArgb(145, 123, 123);
            var colStr1 = serializer.Serialize(col1);

            const string expectedCol1 = @"<Color>
  <A>255</A>
  <R>145</R>
  <G>123</G>
  <B>123</B>
</Color>";

            Assert.That(colStr1, Is.EqualTo(expectedCol1));

            var col2 = SystemColors.ButtonFace;
            var colStr2 = serializer.Serialize(col2);
            const string expectedCol2 = @"<Color>ButtonFace</Color>";

            Assert.That(colStr2, Is.EqualTo(expectedCol2));
        }

        [Test]
        public void TestSerializingNDeserializingNullKnownTypes()
        {
            var inst = ClassContainingXElement.GetSampleInstance();
            inst.TheElement = null;
            inst.TheAttribute = null;

            var ser = new YAXSerializer(typeof (ClassContainingXElement), YAXExceptionHandlingPolicies.ThrowErrorsOnly,
                                        YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);

            try
            {
                var xml = ser.Serialize(inst);
                var deseredInstance = ser.Deserialize(xml);
                Assert.That(deseredInstance.ToString(), Is.EqualTo(inst.ToString()));
            }
            catch (Exception ex)
            {
                Assert.Fail("No exception should have been throwned, but received:\r\n" + ex);
            }

        }

        [Test]
        public void RectangleSerializationTest()
        {
            const string result =
@"<RectangleDynamicKnownTypeSample>
  <Rect>
    <Left>10</Left>
    <Top>20</Top>
    <Width>30</Width>
    <Height>40</Height>
  </Rect>
</RectangleDynamicKnownTypeSample>";
            var serializer = new YAXSerializer(typeof(RectangleDynamicKnownTypeSample), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(RectangleDynamicKnownTypeSample.GetSampleInstance());
            Assert.That(got, Is.EqualTo(result));
        }

        [Test]
        public void DataSetAndDataTableSerializationTest()
        {
            const string result =
@"<DataSetAndDataTableKnownTypeSample>
  <TheDataTable>
    <NewDataSet>
      <TableName xmlns=""http://tableNs/"">
        <Col1>1</Col1>
        <Col2>2</Col2>
        <Col3>3</Col3>
      </TableName>
      <TableName xmlns=""http://tableNs/"">
        <Col1>y</Col1>
        <Col2>4</Col2>
        <Col3>n</Col3>
      </TableName>
    </NewDataSet>
  </TheDataTable>
  <TheDataSet>
    <MyDataSet>
      <Table1>
        <Cl1>num1</Cl1>
        <Cl2>34</Cl2>
      </Table1>
      <Table1>
        <Cl1>num2</Cl1>
        <Cl2>54</Cl2>
      </Table1>
      <Table2>
        <C1>one</C1>
        <C2>1</C2>
        <C3>1.5</C3>
      </Table2>
      <Table2>
        <C1>two</C1>
        <C2>2</C2>
        <C3>2.5</C3>
      </Table2>
    </MyDataSet>
  </TheDataSet>
</DataSetAndDataTableKnownTypeSample>";

            var serializer = new YAXSerializer(typeof(DataSetAndDataTableKnownTypeSample), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(DataSetAndDataTableKnownTypeSample.GetSampleInstance());
            Assert.That(got, Is.EqualTo(result));
        }

    }
}
