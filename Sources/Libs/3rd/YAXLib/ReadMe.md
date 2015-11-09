# YAXLib: Yet Another XML Serialization Library for the .NET Framework

*YAXLib* is a flexible XML serialization library that lets developers design freely the XML file structure, choose among private and public fields to be serialized, and serialize all known collection classes and arrays (single-dimensional, multi-dimensional, and jagged arrays) in the .NET Framework. 

*YAXLib* can be used as a powerful XML parser or generator, that exploits the object to XML mapping in the class definitions themselves. 

The exception handling policies and its missing data recovery capabilities makes *YAXLib* a reliable tool for XML configuration storage and retrieval.

## Features

* Allowing the programmer to format the XML result as desired
* Support for specifying path-like serialization addresses, e.g., `elem1/elem2/elem3`, and `../elem1`, and `./elem1`
* Support for XML namespaces
* Serialization and deserialization of all known generic and non-generic collection classes in `System.Collections`, and `System.Collections.Generic`
* Support for serialization of single-dimensional, multi-dimensional, and jagged arrays
* Support for recursive serialization of collections (i.e., collection of collections)
* Support for specifying aliases for enum members
* Support for defining user-defined custom serializer for specific types or specific fields
* Allowing the programmer to choose the fields to serialize (public, or non-public properties, or member variables)
* Support for serialization and deserialization of objects through a reference to their base-class or interface (also known as polymorphic serialization)
* Support for multi-stage deserialization
* Allowing the programmer to add comments for the elements in the XML result
* and more ...

See the accompanied demo application for an example of each functionality. 

## Documentation

The best documentation for *YAXLib* is its various samples and unit-tests. However this (rather old) [CodeProject article](http://www.codeproject.com/Articles/34045/Yet-Another-XML-Serialization-Library-for-the-NET) is a good source to start. The article is also found in the `Doc` folder of the source code.

To play with sample classes, open one of the solution files (based on version of your Visual Studio), go to `YAXLibTests` project, `SampleClasses` folder. If you want a sample class to appear in the demo application simply put a `[ShowInDemoApplication]` attribute on top of the class definition. In the demo application you can see the serialization result, modify it, and test its deserialization.

## Nuget

To install *YAXLib*, run the following command in the *Package Manager Console*:

    PM> Install-Package YAXLib

## A Quick Introduction

Imagine that we have a simple `Warehouse` class with the following definition:

```csharp
public class Warehouse
{
    public class Person
    {
        public string SSN { get; set; }
        public string Name { get; set; }
        public string Family { get; set; }
        public int Age { get; set; }
    }

    public string Name { get; set; }
    public string Address { get; set; }
    public double Area { get; set; }
    public List<string> Items { get; set; }
    public Dictionary<string, int> ItemQuantitiesDic { get; set; }
    public Person Owner { get; set; }
}
```

Without adding any attributes, *YAXLib* is perfectly capable of serializing objects of the above class. The following is an XML serialization of a sample instantiation of our `Warehouse` class:

```xml
<Warehouse>
  <Name>Foo Warehousing Ltd.</Name>
  <Address>No. 10, Some Ave., Some City, Some Country</Address>
  <Area>120000.5</Area>
  <Items>
    <String>Bicycle</String>
    <String>Football</String>
    <String>Shoe</String>
    <String>Treadmill</String>
  </Items>
  <ItemQuantitiesDic>
    <KeyValuePairOfStringInt32>
      <Key>Bicycle</Key>
      <Value>10</Value>
    </KeyValuePairOfStringInt32>
    <KeyValuePairOfStringInt32>
      <Key>Football</Key>
      <Value>120</Value>
    </KeyValuePairOfStringInt32>
    <KeyValuePairOfStringInt32>
      <Key>Shoe</Key>
      <Value>600</Value>
    </KeyValuePairOfStringInt32>
    <KeyValuePairOfStringInt32>
      <Key>treadmill</Key>
      <Value>25</Value>
    </KeyValuePairOfStringInt32>
  </ItemQuantitiesDic>
  <Owner>
    <SSN>123456789</SSN>
    <Name>John</Name>
    <Family>Doe</Family>
    <Age>50</Age>
  </Owner>
</Warehouse>
```

It's good to have it serialized this simple, but *YAXLib* can do more than that. Let's do some house-keeping on the XML side. Let's move the XML-elements and attributes around, so that we will have a nicer and more human readable XML file. Let's decorate our `Warehouse` class with the following *YAXLib* attributes and see the XML result.

```csharp
[YAXComment("Watch it closely. It's awesome, isn't it!")]
public class Warehouse
{
    public class Person
    {
        [YAXAttributeFor("..")]
        [YAXSerializeAs("OwnerSSN")]
        public string SSN { get; set; }

        [YAXAttributeFor("Identification")]
        public string Name { get; set; }

        [YAXAttributeFor("Identification")]
        public string Family { get; set; }

        public int Age { get; set; }
    }

    [YAXAttributeForClass]
    public string Name { get; set; }

    [YAXSerializeAs("address")]
    [YAXAttributeFor("SiteInfo")]
    public string Address { get; set; }

    [YAXSerializeAs("SurfaceArea")]
    [YAXElementFor("SiteInfo")]
    public double Area { get; set; }

    [YAXCollection(YAXCollectionSerializationTypes.Serially, SeparateBy = ", ")]
    [YAXSerializeAs("AvailableItems")]
    public List<string> Items { get; set; }

    [YAXDictionary(EachPairName = "ItemInfo", KeyName = "Item", ValueName = "Count",
                    SerializeKeyAs = YAXNodeTypes.Attribute,
                    SerializeValueAs = YAXNodeTypes.Attribute)]
    [YAXCollection(YAXCollectionSerializationTypes.RecursiveWithNoContainingElement)]
    [YAXSerializeAs("ItemQuantities")]
    public Dictionary<string, int> ItemQuantitiesDic { get; set; }

    public Person Owner { get; set; }
}
```

And this is the result of XML serialization:

```xml
<!-- Watch it closely. It's awesome, isn't it! -->
<Warehouse Name="Foo Warehousing Ltd." OwnerSSN="123456789">
  <SiteInfo address="No. 10, Some Ave., Some City, Some Country">
    <SurfaceArea>120000.5</SurfaceArea>
  </SiteInfo>
  <AvailableItems>Bicycle, Football, Shoe, Treadmill</AvailableItems>
  <ItemInfo Item="Bicycle" Count="10" />
  <ItemInfo Item="Football" Count="120" />
  <ItemInfo Item="Shoe" Count="600" />
  <ItemInfo Item="treadmill" Count="25" />
  <Owner>
    <Identification Name="John" Family="Doe" />
    <Age>50</Age>
  </Owner>
</Warehouse>
```

## Contact

YAXLib is hosted on both [GitHub](https://github.com/sinairv/YAXLib) and [CodePlex](http://yaxlib.codeplex.com). Feel free to discuss about and fork it on either of these sites that you prefer. 

Copyright (c) 2009 - 2013 Sina Iravanian and Contributors - Licenced under MIT 

Homepage: [www.sinairv.com](http://www.sinairv.com)

Github: [github.com/sinairv](https://github.com/sinairv)

Twitter: [@sinairv](http://www.twitter.com/sinairv)
