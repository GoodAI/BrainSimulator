using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;
using YAXLibTests.SampleClasses;

namespace YAXLibTests
{
    [TestFixture]
    public class OverridingYAXLibMetadataTests
    {
        [Test]
        public void CanOverrideAllMetadata()
        {
            var ser = new YAXSerializer(typeof(YAXLibMetadataOverriding));

            ser.YaxLibNamespacePrefix = "yax";
            ser.YaxLibNamespaceUri = "http://namespace.org/yax";
            ser.DimentionsAttributeName = "dm";
            ser.RealTypeAttributeName = "type";

            var sampleInstance = YAXLibMetadataOverriding.GetSampleInstance();
            string result = ser.Serialize(sampleInstance);

            string expected =
@"<YAXLibMetadataOverriding xmlns:yax=""http://namespace.org/yax"" xmlns=""http://namespace.org/sample"">
  <IntArray yax:dm=""2,3"">
    <Int32>1</Int32>
    <Int32>2</Int32>
    <Int32>3</Int32>
    <Int32>2</Int32>
    <Int32>3</Int32>
    <Int32>4</Int32>
  </IntArray>
  <Obj yax:type=""System.String"">Hello, World!</Obj>
</YAXLibMetadataOverriding>";
            Assert.That(result, Is.EqualTo(expected));
            
            var desObj = (YAXLibMetadataOverriding)ser.Deserialize(expected);

            Assert.That(desObj.Obj.ToString(), Is.EqualTo(sampleInstance.Obj.ToString()));
            Assert.That(desObj.IntArray.Length, Is.EqualTo(sampleInstance.IntArray.Length));
        }

        [Test]
        public void CanUseTheDefaultNamespace()
        {
            var ser = new YAXSerializer(typeof(YAXLibMetadataOverriding));

            ser.YaxLibNamespacePrefix = String.Empty;
            ser.YaxLibNamespaceUri = "http://namespace.org/sample";
            ser.DimentionsAttributeName = "dm";
            ser.RealTypeAttributeName = "type";

            var sampleInstance = YAXLibMetadataOverriding.GetSampleInstance();
            string result = ser.Serialize(sampleInstance);


            string expected =
@"<YAXLibMetadataOverriding xmlns=""http://namespace.org/sample"">
  <IntArray dm=""2,3"">
    <Int32>1</Int32>
    <Int32>2</Int32>
    <Int32>3</Int32>
    <Int32>2</Int32>
    <Int32>3</Int32>
    <Int32>4</Int32>
  </IntArray>
  <Obj type=""System.String"">Hello, World!</Obj>
</YAXLibMetadataOverriding>";
            Assert.That(result, Is.EqualTo(expected));

            var desObj = (YAXLibMetadataOverriding)ser.Deserialize(expected);

            Assert.That(desObj.Obj.ToString(), Is.EqualTo(sampleInstance.Obj.ToString()));
            Assert.That(desObj.IntArray.Length, Is.EqualTo(sampleInstance.IntArray.Length));
        }

    }
}
