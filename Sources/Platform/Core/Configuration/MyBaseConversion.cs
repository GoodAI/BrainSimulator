using GoodAI.Core.Utils;
using System;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Xml.Linq;

namespace GoodAI.Core.Configuration
{
    public abstract class MyBaseConversion
    {
        public abstract int CurrentVersion { get; }        
        public MyModuleConfig Module { get; set; }

        public string ApplyConversionsIfNeeed(string xml, int xmlVersion)
        {
            if (xmlVersion < CurrentVersion)
            {
                MyLog.INFO.WriteLine(Module.File.Name + ": Conversion is needed (stored version: " + xmlVersion + ", current version: " + CurrentVersion + ").");

                Type type = this.GetType();
                bool errorOccured = false;

                string convertedXml = xml;

                for (int i = xmlVersion; i < CurrentVersion; i++)
                {
                    string methodName = "Convert" + i + "To" + (i + 1);
                    MethodInfo methodInfo = type.GetMethod(methodName);

                    if (methodInfo != null)
                    {
                        convertedXml = (string)methodInfo.Invoke(null, new object[] { convertedXml });
                        MyLog.INFO.WriteLine("Conversion to version " + (i + 1) + " finished.");
                    }
                    else
                    {
                        MyLog.ERROR.WriteLine("Conversion between versions " + i + " and " + (i + 1) + " failed: Method \"" + methodName + "\" is missing in (CustomModels)\\Conversions.MyConversions.");
                        errorOccured = true;
                    }
                }

                if (errorOccured)
                {
                    MyLog.INFO.WriteLine(Module.File.Name + ": Automatic conversion failed. Project might be in inconsistent state.");
                }
                else
                {
                    MyLog.INFO.WriteLine(Module.File.Name + ": Automatic conversion finished successfuly. Save is needed.");                    
                }

                return convertedXml;
            }
            else if (xmlVersion > CurrentVersion)
            {
                MyLog.WARNING.WriteLine(Module.File.Name + ": Project used newer version of module.");                
            }

            return xml;
        }

        //TODO: not safe enough
        public static Regex FILE_VERSION_PATTERN = new Regex("<Project Name=\\\".+\\\" FileVersion=\\\"\\d+\\\"");

        public static string ConvertOldModuleNames(string xml)
        {

            string result = xml.Replace("BrainSimulator.dll", MyConfiguration.CORE_MODULE_NAME);
            result = result.Replace("CustomModels.dll", MyConfiguration.BASIC_NODES_MODULE_NAME);
            result = result.Replace("\"MNIST.dll", "\"GoodAI.MNIST.dll");
            result = result.Replace("\"XmlFeedForwardNet.dll", "\"GoodAI.XmlFeedForwardNet.dll");

            if (!xml.Equals(result))
            {
                MyLog.INFO.WriteLine("Old module names found and converted. Save is needed.");
            }

            return result;
        }

        public static string ConvertOldFileVersioning(string xml)
        {
            Match m = FILE_VERSION_PATTERN.Match(xml);

            int fileVersion = 1;

            if (m.Success)
            {
                MyLog.INFO.WriteLine("Old document versioning found. Converting...");

                fileVersion = int.Parse(m.Value.Split('"')[3]);

                XDocument doc = XDocument.Parse(xml);
                doc.Root.Add(XElement.Parse(
                    "<UsedModules>" +
                        "<Module Name=\"" + MyConfiguration.CORE_MODULE_NAME + "\" Version=\"" + fileVersion + "\"></Module>" +
                        "<Module Name=\"" + MyConfiguration.BASIC_NODES_MODULE_NAME + "\" Version=\"1\"></Module>" +
                    "</UsedModules>"));

                return doc.ToString();
            }
            else return xml;
        }
    }
}
