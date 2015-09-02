using GoodAI.Core.Utils;
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Xml.Linq;

namespace GoodAI.BrainSimulator.Utils
{
    public class MyDocProvider
    {
        private const string XMLDOC_FILENAME = "doc.xml";
        private Dictionary<string, XElement> m_docTable = new Dictionary<string, XElement>();

        public void LoadXMLDoc(Assembly assembly)
        {
            try
            {
                string src = MyResources.GetTextFromAssembly(assembly, XMLDOC_FILENAME);
                
                XDocument xml = XDocument.Parse(src);
                IEnumerable<XElement> members = xml.Root.Element("members").Elements("member");

                foreach (XElement member in members)                    
                {
                    m_docTable[member.Attribute("name").Value] = member;
                }
            }
            catch (Exception ex)
            {
                MyLog.WARNING.WriteLine("XML documentation loading failed for assembly: " + assembly.FullName + "\n\t" + ex.Message);
            }
        }

        private static string GenerateMemberName<T>(T member)
        {
            if (member is Type)
            {
                return "T:" + (member as Type).FullName.Replace('+', '.');
            }
            else return "NO_ELEMENT";            
        }

        private static string StripWhiteSpace(string htmlText)
        {
            Regex spaces = new Regex(" +", RegexOptions.IgnoreCase);
            string result = spaces.Replace(htmlText, " ").Replace("\n ", "\n");

            return result;
        }

        public bool HasDocElement<T>(T type, string elementName, out string result)
        {
            string memberName = GenerateMemberName(type);
            result = null;

            if (m_docTable.ContainsKey(memberName))
            {
                XElement element = m_docTable[memberName].Element(elementName);
                if (element != null)
                {
                    var reader = element.CreateReader();
                    reader.MoveToContent();

                    result = WebUtility.HtmlDecode(reader.ReadInnerXml().Trim());
                    result = StripWhiteSpace(result);
                }
            }

            return result != null;
        }

        public bool HasSummary(Type type, out string result)
        {
            return HasDocElement(type, "summary", out result);
        }

        public bool HasDescription(Type type, out string result)
        {
            return HasDocElement(type, "description", out result);
        }

        public bool HasStatus(Type type, out string result)
        {
            return HasDocElement(type, "status", out result);
        }

        public bool HasAuthor(Type type, out string result)
        {
            return HasDocElement(type, "author", out result);
        }

        private static string GetDocumentationPath(Type forType)
        {
            return Directory.GetCurrentDirectory() + "\\doc\\" + forType.Namespace.Replace('.', '\\') + "\\" + forType.Name + ".html";            
        }

        private static string GetDefaultBrowserPath()
        {
            string key = @"HTTP\shell\open\command";
            using (RegistryKey registrykey = Registry.ClassesRoot.OpenSubKey(key, false))
            {
                return ((string)registrykey.GetValue(null, null)).Split('"')[1];
            }
        }

        public static void Navigate(string url)
        {
            Process.Start(GetDefaultBrowserPath(), url);
        }

        public static void NavigateToDoc(Type forType)
        {
            string fileName = GetDocumentationPath(forType);

            if (File.Exists(fileName))
            {
                Process.Start(GetDefaultBrowserPath(), "file:///" + fileName);
            }
            else
            {
                MyLog.ERROR.WriteLine("Documentation for \"" + forType.Name + "\" not found.");
            }
        }
    }
}
