using GoodAI.Core.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace GoodAI.Modules.Versioning
{
    public class MyConversion : MyBaseConversion
    {
        public override int CurrentVersion { get { return 4; } }


        /// <summary>
        /// Convert RandomMapper task name and property names
        /// Author: Martin Milota
        /// </summary>
        public static string Convert1To2(string xml)
        {
            xml = xml.Replace("MyRandomMappertask", "MyRandomMapperTask");


            XDocument document = XDocument.Parse(xml);

            if (document.Root == null)
                return xml;


            List<XElement> toRemove = new List<XElement>();

            foreach (var mapper in document.Root.Descendants("MyRandomMapper"))
            {
                // Move the DoDecoding property from the only task to the node
                {
                    string doDecoding = "false";

                    foreach (var task in mapper.Descendants("Task"))
                    {
                        if (task.Attributes().Any(prop => prop.Value.Contains("MyRandomMapperTask")))
                        {
                            foreach (var decoding in task.Descendants("DoDecoding"))
                            {
                                doDecoding = decoding.Value;
                                toRemove.Add(decoding);
                            }
                        }
                    }

                    mapper.AddFirst(new XElement("DoDecoding", doDecoding));
                }


                // Add a new task that is enabled by default
                foreach (var tasks in mapper.Descendants("Tasks"))
                {
                    XElement generateTask = new XElement("Task");
                    generateTask.SetAttributeValue("Enabled", "True");
                    generateTask.SetAttributeValue("PropertyName", "InitTask");

                    XNamespace yaxlib = "http://www.sinairv.com/yaxlib/";
                    XName realtype = yaxlib + "realtype";
                    generateTask.SetAttributeValue(realtype, "BrainSimulator.VSA.MyRandomMapper+MyGenerateMatrixTask");

                    generateTask.SetElementValue("AxisToNormalize", "yDim");

                    tasks.AddFirst(generateTask);
                }


                // Replace the Orthonormalize property by VectorMode
                foreach (var ortho in mapper.Descendants("Orthonormalize"))
                {
                    ortho.AddBeforeSelf(new XElement("VectorMode", ortho.Value == "True" ? "Orthonormalize" : "Normal"));
                    toRemove.Add(ortho);
                }
            }


            foreach (var xElement in toRemove)
            {
                xElement.Remove();
            }

            return document.ToString();
        }

        /// <summary>
        /// Big fat conversion of old namespace names
        /// Author: Dusan Fedorcak
        /// </summary>
        public static string Convert2To3(string xml)
        {
            string result = xml;

            result = result.Replace("yaxlib:realtype=\"BrainSimulator.Nodes", "yaxlib:realtype=\"GoodAI.Core.Nodes");            
            
            result = result.Replace("yaxlib:realtype=\"BrainSimulator.Observers.MyHistogramObserver", "yaxlib:realtype=\"GoodAI.Core.Observers.MyHistogramObserver");
            result = result.Replace("yaxlib:realtype=\"BrainSimulator.Observers.MyMatrixObserver", "yaxlib:realtype=\"GoodAI.Core.Observers.MyMatrixObserver");
            result = result.Replace("yaxlib:realtype=\"BrainSimulator.Observers.MyMemoryBlockObserver", "yaxlib:realtype=\"GoodAI.Core.Observers.MyMemoryBlockObserver");
            result = result.Replace("yaxlib:realtype=\"BrainSimulator.Observers.MySpikeRasterObserver", "yaxlib:realtype=\"GoodAI.Core.Observers.MySpikeRasterObserver");
            result = result.Replace("yaxlib:realtype=\"BrainSimulator.Observers.MyTimePlotObserver", "yaxlib:realtype=\"GoodAI.Core.Observers.MyTimePlotObserver");

            result = result.Replace("yaxlib:realtype=\"BrainSimulator.", "yaxlib:realtype=\"GoodAI.Modules.");

            return result;
        }

        /// <summary>
        /// Move random node to proper namespace
        /// Author: MB
        /// </summary>
        public static string Convert3To4(string xml)
        {
            string result = xml;

            result = result.Replace("yaxlib:realtype=\"GoodAI.Modules.Testing.MyRandomNode", "yaxlib:realtype=\"GoodAI.Modules.Common.MyRandomNode");

            return result;
        }

    }
}
