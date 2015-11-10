using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    public class CsprojParser
    {
        public static ProjectBuildDefinition Parse(string xml)
        {
            var yaxSer = new YAXSerializer(typeof(ProjectBuildDefinition),
                YAXExceptionHandlingPolicies.DoNotThrow,
                YAXExceptionTypes.Ignore,
                YAXSerializationOptions.DontSerializeNullObjects);

            return yaxSer.Deserialize(xml) as ProjectBuildDefinition;
        }

        public static string ParseAndRegenerateXml(string xml)
        {
            var project = Parse(xml);

            var yaxSer = new YAXSerializer(typeof(ProjectBuildDefinition),
                YAXExceptionHandlingPolicies.DoNotThrow,
                YAXExceptionTypes.Ignore,
                YAXSerializationOptions.DontSerializeNullObjects);

            return yaxSer.Serialize(project);
        }
    }

    [YAXNamespace("http://schemas.microsoft.com/developer/msbuild/2003")]
    [YAXSerializeAs("Project")]
    public class ProjectBuildDefinition
    {
        [YAXAttributeForClass]
        public string ToolsVersion { get; set; }

        [YAXAttributeForClass]
        public string DefaultTargets { get; set; }

        [YAXCollection(YAXCollectionSerializationTypes.RecursiveWithNoContainingElement,
            EachElementName = "PropertyGroup")]
        public List<PropertyGroup> PropertyGroups { get; set; }

        [YAXCollection(YAXCollectionSerializationTypes.RecursiveWithNoContainingElement,
           EachElementName = "ItemGroup")]
        public List<ItemGroup> ItemGroups { get; set; }

        [YAXCollection(YAXCollectionSerializationTypes.RecursiveWithNoContainingElement,
           EachElementName = "Import")]
        public List<ImportItem> ImportItems { get; set; }
    }

    public class PropertyGroup
    {
        public string Configuration { get; set; }

        [YAXSerializeAs("Condition")]
        [YAXAttributeFor("Configuration")]
        public string ConfigCondition { get; set; }


        public string Platform { get; set; }

        [YAXSerializeAs("Condition")]
        [YAXAttributeFor("Platform")]
        public string PlatformCondition { get; set; }

        public string ProductVersion { get; set; }
        public string SchemaVersion { get; set; }
        public string ProjectGuid { get; set; }
        public string OutputType { get; set; }
        public string AppDesignerFolder { get; set; }
        public string RootNamespace { get; set; }
        public string AssemblyName { get; set; }
        public string TargetFrameworkVersion { get; set; }
        public string FileAlignment { get; set; }

        public bool DebugSymbols { get; set; }
        public string DebugType { get; set; }
        public bool Optimize { get; set; }
        public string OutputPath { get; set; }
        public string DefineConstants { get; set; }
        public string ErrorReport { get; set; }
        public int WarningLevel { get; set; }
        public string DocumentationFile { get; set; }
        public string PostBuildEvent { get; set; }
    }


    public class ItemGroup
    {
        [YAXCollection(YAXCollectionSerializationTypes.RecursiveWithNoContainingElement, 
            EachElementName = "Reference")]
        public List<ReferenceItem> ReferenceItems { get; set; }
    }


    [YAXSerializeAs("Reference")]
    public class ReferenceItem
    {
        [YAXAttributeForClass]
        public string Include { get; set; }

        public string HintPath { get; set; }
        public string RequiredTargetFramework { get; set; }
        public bool SpecificVersion { get; set; }
    }

    [YAXSerializeAs("Import")]
    public class ImportItem
    {
        [YAXAttributeForClass()]
        public string Project { get; set; }
    }

}
