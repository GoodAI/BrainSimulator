using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Linq;

namespace YAXLibTests.SampleClasses
{
    public class ClassContainingXElement
    {
        public XElement TheElement { get; set; }
        public XAttribute TheAttribute { get; set; }


        public override string ToString()
        {
            return String.Format("TheElement: {0}\r\nTheAttribute: {1}\r\n",
                TheElement, TheAttribute);
        }

        public static ClassContainingXElement GetSampleInstance()
        {
            var elem = new XElement("SomeElement", 
                new XElement("Child", "Content"),
                new XElement("Multi-level", 
                    new XElement("GrandChild", "Content")),
                new XAttribute("someattribute", "value"));

            var attrib = new XAttribute("attrib", "TheAttribValue");

            return new ClassContainingXElement {TheElement = elem, TheAttribute = attrib};
        }
    }
}
