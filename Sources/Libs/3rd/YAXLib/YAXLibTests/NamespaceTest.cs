// Copyright 2009 - 2010 Sina Iravanian - <sina@sinairv.com>
//
// This source file(s) may be redistributed, altered and customized
// by any means PROVIDING the authors name and all copyright
// notices remain intact.
// THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED. USE IT AT YOUR OWN RISK. THE AUTHOR ACCEPTS NO
// LIABILITY FOR ANY DATA DAMAGE/LOSS THAT THIS PRODUCT MAY CAUSE.
//-----------------------------------------------------------------------

using NUnit.Framework;

using YAXLib;

using YAXLibTests.SampleClasses;
using YAXLibTests.SampleClasses.Namespace;

namespace YAXLibTests
{
    [TestFixture]
    public class NamespaceTest
    {
        [Test]
        public void SingleNamespaceSerializationTest()
        {
            const string result = @"<!-- This example shows usage of a custom default namespace -->
" + "<SingleNamespaceSample xmlns=\"http://namespaces.org/default\">" + @"
  <StringItem>This is a test string</StringItem>
  <IntItem>10</IntItem>
</SingleNamespaceSample>";

            var serializer = new YAXSerializer(typeof(SingleNamespaceSample), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(SingleNamespaceSample.GetInstance());
            Assert.That(got, Is.EqualTo(result));
        }

        [Test]
        public void MultipleNamespaceSerializationTest()
        {
            const string result = 
@"<!-- This example shows usage of a number of custom namespaces -->
<ns1:MultipleNamespaceSample xmlns:ns1=""http://namespaces.org/ns1"" xmlns:ns2=""http://namespaces.org/ns2"" xmlns:ns3=""http://namespaces.org/ns3"">
  <ns1:BoolItem>True</ns1:BoolItem>
  <ns2:StringItem>This is a test string</ns2:StringItem>
  <ns3:IntItem>10</ns3:IntItem>
</ns1:MultipleNamespaceSample>";

            var serializer = new YAXSerializer(typeof(MultipleNamespaceSample), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(MultipleNamespaceSample.GetSampleInstance());
            Assert.That(got, Is.EqualTo(result));
        }

        [Test]
        public void AttributeNamespaceSerializationTest()
        {
            const string result = 
@"<AttributeNamespaceSample xmlns:ns=""http://namespaces.org/ns"" xmlns=""http://namespaces.org/default"">
  <Attribs attrib=""value"" ns:attrib2=""value2"" />
</AttributeNamespaceSample>";

            var serializer = new YAXSerializer(typeof(AttributeNamespaceSample), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(AttributeNamespaceSample.GetSampleInstance());
            Assert.That(got, Is.EqualTo(result));
        }

        [Test]
        public void MemberAndClassDifferentNamespacesDeserializationTest()
        {
            const string result = @"<CellPhone_MemberAndClassDifferentNamespaces xmlns:x1=""http://namespace.org/x1"" xmlns=""http://namespace.org/nsmain"">
  <x1:TheName>HTC</x1:TheName>
  <OS>Windows Phone 8</OS>
</CellPhone_MemberAndClassDifferentNamespaces>";

            var serializer = new YAXSerializer(typeof(CellPhone_MemberAndClassDifferentNamespaces), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(CellPhone_MemberAndClassDifferentNamespaces.GetSampleInstance());
            Assert.That(got, Is.EqualTo(result));
        }

        [Test]
        public void MemberAndClassDifferentNamespacePrefixesSerializationTest()
        {
            const string result = 
@"<xmain:CellPhone_MemberAndClassDifferentNamespacePrefixes xmlns:xmain=""http://namespace.org/nsmain"" xmlns:x1=""http://namespace.org/x1"">
  <x1:TheName>HTC</x1:TheName>
  <xmain:OS>Windows Phone 8</xmain:OS>
</xmain:CellPhone_MemberAndClassDifferentNamespacePrefixes>";

            var serializer = new YAXSerializer(typeof(CellPhone_MemberAndClassDifferentNamespacePrefixes), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(CellPhone_MemberAndClassDifferentNamespacePrefixes.GetSampleInstance());
            Assert.That(got, Is.EqualTo(result));
        }

        [Test]
        public void MultiLevelMemberAndClassDifferentNamespacesSerializationTest()
        {
            const string result =
@"<CellPhone_MultiLevelMemberAndClassDifferentNamespaces xmlns:x1=""http://namespace.org/x1"" xmlns=""http://namespace.org/nsmain"">
  <Level1>
    <Level2>
      <x1:TheName>HTC</x1:TheName>
    </Level2>
  </Level1>
  <OS>Windows Phone 8</OS>
</CellPhone_MultiLevelMemberAndClassDifferentNamespaces>";

            var serializer = new YAXSerializer(typeof(CellPhone_MultiLevelMemberAndClassDifferentNamespaces), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(CellPhone_MultiLevelMemberAndClassDifferentNamespaces.GetSampleInstance());
            Assert.That(got, Is.EqualTo(result));
        }

        [Test]
        public void DictionaryNamespaceSerializationTest()
        {
            const string result = 
@"<CellPhone_DictionaryNamespace xmlns:x1=""http://namespace.org/x1"" xmlns:p1=""namespace/for/prices/only"" xmlns=""http://namespace.org/nsmain"">
  <x1:TheName>HTC</x1:TheName>
  <OS>Windows Phone 8</OS>
  <p1:Prices>
    <p1:KeyValuePairOfColorDouble>
      <p1:Key>Red</p1:Key>
      <p1:Value>120</p1:Value>
    </p1:KeyValuePairOfColorDouble>
    <p1:KeyValuePairOfColorDouble>
      <p1:Key>Blue</p1:Key>
      <p1:Value>110</p1:Value>
    </p1:KeyValuePairOfColorDouble>
    <p1:KeyValuePairOfColorDouble>
      <p1:Key>Black</p1:Key>
      <p1:Value>140</p1:Value>
    </p1:KeyValuePairOfColorDouble>
  </p1:Prices>
</CellPhone_DictionaryNamespace>";
            var serializer = new YAXSerializer(typeof(CellPhone_DictionaryNamespace), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(CellPhone_DictionaryNamespace.GetSampleInstance());
            Assert.That(got, Is.EqualTo(result));
        }

        [Test]
        public void DictionaryNamespaceForAllItemsSerializationTest()
        {
            const string result =
@"<CellPhone_DictionaryNamespaceForAllItems xmlns:p1=""http://namespace.org/brand"" xmlns:p2=""http://namespace.org/prices"" xmlns:p3=""http://namespace.org/pricepair"" xmlns:p4=""http://namespace.org/color"" xmlns:p5=""http://namespace.org/pricevalue"">
  <p1:Brand>Samsung Galaxy Nexus</p1:Brand>
  <OS>Android</OS>
  <p2:ThePrices>
    <p3:PricePair>
      <p4:TheColor>Red</p4:TheColor>
      <p5:ThePrice>120</p5:ThePrice>
    </p3:PricePair>
    <p3:PricePair>
      <p4:TheColor>Blue</p4:TheColor>
      <p5:ThePrice>110</p5:ThePrice>
    </p3:PricePair>
    <p3:PricePair>
      <p4:TheColor>Black</p4:TheColor>
      <p5:ThePrice>140</p5:ThePrice>
    </p3:PricePair>
  </p2:ThePrices>
</CellPhone_DictionaryNamespaceForAllItems>";
            var serializer = new YAXSerializer(typeof(CellPhone_DictionaryNamespaceForAllItems), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(CellPhone_DictionaryNamespaceForAllItems.GetSampleInstance());
            Assert.That(got, Is.EqualTo(result));
        }

        [Test]
        public void CollectionNamespaceGoesThruRecursiveNoContainingElementSerializationTest()
        {
            const string result =
@"<MobilePhone xmlns:app=""http://namespace.org/apps"">
  <DeviceBrand>Samsung Galaxy Nexus</DeviceBrand>
  <OS>Android</OS>
  <app:String>Google Map</app:String>
  <app:String>Google+</app:String>
  <app:String>Google Play</app:String>
</MobilePhone>";
            var serializer = new YAXSerializer(typeof(CellPhone_CollectionNamespaceGoesThruRecursiveNoContainingElement), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(CellPhone_CollectionNamespaceGoesThruRecursiveNoContainingElement.GetSampleInstance());
            Assert.That(got, Is.EqualTo(result));
        }

        [Test]
        public void CollectionNamespaceForAllItemsSerializationTest()
        {
            const string result =
@"<MobilePhone xmlns:app=""http://namespace.org/apps"" xmlns:cls=""http://namespace.org/colorCol"" xmlns:mdls=""http://namespace.org/modelCol"" xmlns:p1=""http://namespace.org/appName"" xmlns:p2=""http://namespace.org/color"">
  <DeviceBrand>Samsung Galaxy Nexus</DeviceBrand>
  <OS>Android</OS>
  <p1:AppName>Google Map</p1:AppName>
  <p1:AppName>Google+</p1:AppName>
  <p1:AppName>Google Play</p1:AppName>
  <cls:AvailableColors>
    <p2:TheColor>Red</p2:TheColor>
    <p2:TheColor>Black</p2:TheColor>
    <p2:TheColor>White</p2:TheColor>
  </cls:AvailableColors>
  <mdls:AvailableModels>S1,MII,SXi,NoneSense</mdls:AvailableModels>
</MobilePhone>";
            var serializer = new YAXSerializer(typeof(CellPhone_CollectionNamespaceForAllItems), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(CellPhone_CollectionNamespaceForAllItems.GetSampleInstance());
            Assert.That(got, Is.EqualTo(result));
        }

        [Test]
        public void YAXNamespaceOverridesImplicitNamespaceSerializationTest()
        {
            const string result =
@"<CellPhone_YAXNamespaceOverridesImplicitNamespace xmlns:p1=""http://namespace.org/explicitBrand"" xmlns:p2=""http://namespace.org/os"">
  <p1:Brand>Samsung Galaxy S II</p1:Brand>
  <p2:OperatingSystem>Android 2</p2:OperatingSystem>
</CellPhone_YAXNamespaceOverridesImplicitNamespace>";

            var serializer = new YAXSerializer(typeof(CellPhone_YAXNamespaceOverridesImplicitNamespace), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(CellPhone_YAXNamespaceOverridesImplicitNamespace.GetSampleInstance());
            Assert.That(got, Is.EqualTo(result));
        }

        [Test]
        public void MutlilevelObjectsWithNamespacesSerializationTest()
        {
            const string result =
@"<MutlilevelObjectsWithNamespaces xmlns:ch1=""http://namespace.org/ch1"" xmlns:ch2=""http://namespace.org/ch2"" xmlns=""http://namespace.org/default"">
  <Parent1>
    <ch1:Child1 ch2:Value3=""Value3"">
      <ch1:Field1>Field1</ch1:Field1>
      <ch1:Field2>Field2</ch1:Field2>
      <ch2:Value1>Value1</ch2:Value1>
      <ch2:Value2>Value2</ch2:Value2>
    </ch1:Child1>
  </Parent1>
  <Parent2>
    <ch2:Child2>
      <ch2:Value4>Value4</ch2:Value4>
    </ch2:Child2>
  </Parent2>
</MutlilevelObjectsWithNamespaces>";

            var serializer = new YAXSerializer(typeof(MutlilevelObjectsWithNamespaces), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(MutlilevelObjectsWithNamespaces.GetSampleInstance());
            Assert.That(got, Is.EqualTo(result));
        }

        [Test]
        public void DictionaryWithParentNamespaceSerializationTest()
        {
            const string result =
@"<Warehouse_Dictionary xmlns=""http://www.mywarehouse.com/warehouse/def/v3"">
  <ItemInfo Item=""Item1"" Count=""10"" />
  <ItemInfo Item=""Item4"" Count=""30"" />
  <ItemInfo Item=""Item2"" Count=""20"" />
</Warehouse_Dictionary>";
            var serializer = new YAXSerializer(typeof(Warehouse_Dictionary), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(Warehouse_Dictionary.GetSampleInstance());
            Assert.That(got, Is.EqualTo(result));
        }

        [Test]
        public void AttributeWithDefaultNamespaceSerializationTest()
        {
            const string result =
                @"<w:font w:name=""Arial"" xmlns:w=""http://example.com/namespace"" />";

            var serializer = new YAXSerializer(typeof(AttributeWithNamespace), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(AttributeWithNamespace.GetSampleInstance());
            Assert.That(got, Is.EqualTo(result));
        }

        [Test]
        public void AttributeWithDefaultNamespaceAsMemberSerializationTest()
        {
            const string result = 
@"<AttributeWithNamespaceAsMember xmlns:w=""http://example.com/namespace"">
  <w:Member w:name=""Arial"" />
</AttributeWithNamespaceAsMember>";

            var serializer = new YAXSerializer(typeof(AttributeWithNamespaceAsMember), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(AttributeWithNamespaceAsMember.GetSampleInstance());
            Assert.That(got, Is.EqualTo(result));
        }

        [Test]
        public void SingleNamespaceDeserializationTest()
        {            
            var serializer = new YAXSerializer(typeof(SingleNamespaceSample), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string serialized = serializer.Serialize(SingleNamespaceSample.GetInstance());
            var deserialized = serializer.Deserialize(serialized) as SingleNamespaceSample;
            Assert.That(deserialized, Is.Not.Null);
            Assert.That(serializer.ParsingErrors, Has.Count.EqualTo(0));
        }

        [Test]
        public void MultipleNamespaceDeserializationTest()
        {
            var serializer = new YAXSerializer(typeof(MultipleNamespaceSample), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string serialized = serializer.Serialize(MultipleNamespaceSample.GetSampleInstance());
            var deserialized = serializer.Deserialize(serialized) as MultipleNamespaceSample;
            Assert.That(deserialized, Is.Not.Null);
            Assert.That(serializer.ParsingErrors, Has.Count.EqualTo(0));
        }
        
        [Test]
        public void AttributeNamespaceDeserializationTest()
        {
            var serializer = new YAXSerializer(typeof(AttributeNamespaceSample), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(AttributeNamespaceSample.GetSampleInstance());
            var deserialized = serializer.Deserialize(got) as AttributeNamespaceSample;
            Assert.That(deserialized, Is.Not.Null);
            Assert.That(serializer.ParsingErrors, Has.Count.EqualTo(0));
        }

        [Test]
        public void MemberAndClassDifferentNamespacesSerializationTest()
        {
            var serializer = new YAXSerializer(typeof(CellPhone_MemberAndClassDifferentNamespaces), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(CellPhone_MemberAndClassDifferentNamespaces.GetSampleInstance());
            var deserialized = serializer.Deserialize(got) as CellPhone_MemberAndClassDifferentNamespaces;
            Assert.That(deserialized, Is.Not.Null);
            Assert.That(serializer.ParsingErrors, Has.Count.EqualTo(0));
        }

        [Test]
        public void MemberAndClassDifferentNamespacePrefixesDeserializationTest()
        {
            var serializer = new YAXSerializer(typeof(CellPhone_MemberAndClassDifferentNamespacePrefixes), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(CellPhone_MemberAndClassDifferentNamespacePrefixes.GetSampleInstance());
            var deserialized = serializer.Deserialize(got) as CellPhone_MemberAndClassDifferentNamespacePrefixes;
            Assert.That(deserialized, Is.Not.Null);
            Assert.That(serializer.ParsingErrors, Has.Count.EqualTo(0));
        }

        [Test]
        public void MultiLevelMemberAndClassDifferentNamespacesDeserializationTest()
        {
            var serializer = new YAXSerializer(typeof(CellPhone_MultiLevelMemberAndClassDifferentNamespaces), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(CellPhone_MultiLevelMemberAndClassDifferentNamespaces.GetSampleInstance());
            var deserialized = serializer.Deserialize(got) as CellPhone_MultiLevelMemberAndClassDifferentNamespaces;
            Assert.That(deserialized, Is.Not.Null);
            Assert.That(serializer.ParsingErrors, Has.Count.EqualTo(0));
        }

        [Test]
        public void DictionaryNamespaceDeserializationTest()
        {
            var serializer = new YAXSerializer(typeof(CellPhone_DictionaryNamespaceForAllItems), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(CellPhone_DictionaryNamespaceForAllItems.GetSampleInstance());
            var deserialized = serializer.Deserialize(got) as CellPhone_DictionaryNamespaceForAllItems;
            Assert.That(deserialized, Is.Not.Null);
            Assert.That(serializer.ParsingErrors, Has.Count.EqualTo(0));
        }

        [Test]
        public void DictionaryNamespaceForAllItemsDeserializationTest()
        {
            var serializer = new YAXSerializer(typeof(CellPhone_DictionaryNamespace), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(CellPhone_DictionaryNamespace.GetSampleInstance());
            var deserialized = serializer.Deserialize(got) as CellPhone_DictionaryNamespace;
            Assert.That(deserialized, Is.Not.Null);
            Assert.That(serializer.ParsingErrors, Has.Count.EqualTo(0));

        }

        [Test]
        public void CollectionNamespaceGoesThruRecursiveNoContainingElementDeserializationTest()
        {
            var serializer = new YAXSerializer(typeof(CellPhone_CollectionNamespaceGoesThruRecursiveNoContainingElement), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(CellPhone_CollectionNamespaceGoesThruRecursiveNoContainingElement.GetSampleInstance());
            var deserialized = serializer.Deserialize(got) as CellPhone_CollectionNamespaceGoesThruRecursiveNoContainingElement;
            Assert.That(deserialized, Is.Not.Null);
            Assert.That(serializer.ParsingErrors, Has.Count.EqualTo(0));
        }

        [Test]
        public void CollectionNamespaceForAllItemsDeserializationTest()
        {
            var serializer = new YAXSerializer(typeof(CellPhone_CollectionNamespaceForAllItems), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(CellPhone_CollectionNamespaceForAllItems.GetSampleInstance());
            var deserialized = serializer.Deserialize(got) as CellPhone_CollectionNamespaceForAllItems;
            Assert.That(deserialized, Is.Not.Null);
            Assert.That(serializer.ParsingErrors, Has.Count.EqualTo(0));
        }

        [Test]
        public void YAXNamespaceOverridesImplicitNamespaceDeserializationTest()
        {
            var serializer = new YAXSerializer(typeof(CellPhone_YAXNamespaceOverridesImplicitNamespace), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(CellPhone_YAXNamespaceOverridesImplicitNamespace.GetSampleInstance());
            var deserialized = serializer.Deserialize(got) as CellPhone_YAXNamespaceOverridesImplicitNamespace;
            Assert.That(deserialized, Is.Not.Null);
            Assert.That(serializer.ParsingErrors, Has.Count.EqualTo(0));
        }

        [Test]
        public void MutlilevelObjectsWithNamespacesDeserializationTest()
        {
            var serializer = new YAXSerializer(typeof(MutlilevelObjectsWithNamespaces), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(MutlilevelObjectsWithNamespaces.GetSampleInstance());
            var deserialized = serializer.Deserialize(got) as MutlilevelObjectsWithNamespaces;
            Assert.That(deserialized, Is.Not.Null);
            Assert.That(serializer.ParsingErrors, Has.Count.EqualTo(0));
        }

        [Test]
        public void DictionaryWithParentNamespaceDeserializationTest()
        {
            var serializer = new YAXSerializer(typeof(Warehouse_Dictionary), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(Warehouse_Dictionary.GetSampleInstance());
            var deserialized = serializer.Deserialize(got) as Warehouse_Dictionary;
            Assert.That(deserialized, Is.Not.Null);
            Assert.That(serializer.ParsingErrors, Has.Count.EqualTo(0));
        }

        [Test]
        public void AttributeWithDefaultNamespaceDeserializationTest()
        {
            var serializer = new YAXSerializer(typeof(AttributeWithNamespace), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(AttributeWithNamespace.GetSampleInstance());
            var deserialized = serializer.Deserialize(got) as AttributeWithNamespace;
            Assert.That(deserialized, Is.Not.Null);
            Assert.That(serializer.ParsingErrors, Has.Count.EqualTo(0));
        }

        [Test]
        public void AttributeWithDefaultNamespaceAsMemberDeserializationTest()
        {
            var serializer = new YAXSerializer(typeof(AttributeWithNamespaceAsMember), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Warning, YAXSerializationOptions.SerializeNullObjects);
            string got = serializer.Serialize(AttributeWithNamespaceAsMember.GetSampleInstance());
            var deserialized = serializer.Deserialize(got) as AttributeWithNamespaceAsMember;
            Assert.That(deserialized, Is.Not.Null);
            Assert.That(serializer.ParsingErrors, Has.Count.EqualTo(0));
        }

        [Test]
        public void CSProjParsingTest()
        {
            string csprojContent = @"<Project ToolsVersion=""4.0"" DefaultTargets=""Build"" xmlns=""http://schemas.microsoft.com/developer/msbuild/2003"">
  <PropertyGroup>
    <Configuration Condition="" '$(Configuration)' == '' "">Debug</Configuration>
    <Platform Condition="" '$(Platform)' == '' "">AnyCPU</Platform>
    <ProductVersion>9.0.30729</ProductVersion>
    <SchemaVersion>2.0</SchemaVersion>
    <ProjectGuid>$guid$</ProjectGuid>
    <OutputType>Library</OutputType>
    <AppDesignerFolder>Properties</AppDesignerFolder>
    <RootNamespace>$safeprojectname$</RootNamespace>
    <AssemblyName>$safeprojectname$</AssemblyName>
    <TargetFrameworkVersion>v4.0</TargetFrameworkVersion>
    <FileAlignment>512</FileAlignment>
    <DebugSymbols>False</DebugSymbols>
    <Optimize>False</Optimize>
    <WarningLevel>0</WarningLevel>
  </PropertyGroup>
  <PropertyGroup>
    <DebugSymbols>True</DebugSymbols>
    <DebugType>full</DebugType>
    <Optimize>False</Optimize>
    <OutputPath>bin\Debug\</OutputPath>
    <DefineConstants>DEBUG;TRACE</DefineConstants>
    <ErrorReport>prompt</ErrorReport>
    <WarningLevel>4</WarningLevel>
    <DocumentationFile>bin\Debug\$safeprojectname$.xml</DocumentationFile>
  </PropertyGroup>
  <PropertyGroup>
    <DebugSymbols>False</DebugSymbols>
    <DebugType>pdbonly</DebugType>
    <Optimize>True</Optimize>
    <OutputPath>bin\Release\</OutputPath>
    <DefineConstants>TRACE</DefineConstants>
    <ErrorReport>prompt</ErrorReport>
    <WarningLevel>4</WarningLevel>
  </PropertyGroup>
  <ItemGroup>
    <Reference Include=""$generatedproject$.EFDAL.Interfaces"">
      <HintPath>..\bin\$generatedproject$.EFDAL.Interfaces.dll</HintPath>
      <SpecificVersion>False</SpecificVersion>
    </Reference>
    <Reference Include=""System"">
      <SpecificVersion>False</SpecificVersion>
    </Reference>
    <Reference Include=""System.Core"">
      <RequiredTargetFramework>3.5</RequiredTargetFramework>
      <SpecificVersion>False</SpecificVersion>
    </Reference>
    <Reference Include=""nHydrate.EFCore, Version=0.0.0.0, Culture=neutral, processorArchitecture=MSIL"">
      <HintPath>..\bin\nHydrate.EFCore.dll</HintPath>
      <SpecificVersion>False</SpecificVersion>
    </Reference>
  </ItemGroup>
  <ItemGroup>
    <Reference Include=""$generatedproject$.EFDAL.Interfaces"">
      <HintPath>..\bin\$generatedproject$.EFDAL.Interfaces.dll</HintPath>
      <SpecificVersion>False</SpecificVersion>
    </Reference>
    <Reference Include=""System"">
      <SpecificVersion>False</SpecificVersion>
    </Reference>
    <Reference Include=""System.Core"">
      <RequiredTargetFramework>3.5</RequiredTargetFramework>
      <SpecificVersion>False</SpecificVersion>
    </Reference>
  </ItemGroup>
  <Import Project=""$(MSBuildToolsPath)\Microsoft.CSharp.targets"" />
</Project>";

            var project = CsprojParser.Parse(csprojContent);
            string xml2 = CsprojParser.ParseAndRegenerateXml(csprojContent);
            Assert.That(xml2, Is.EqualTo(csprojContent));
        }
    }
}
