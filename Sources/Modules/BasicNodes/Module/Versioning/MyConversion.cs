using GoodAI.Core.Configuration;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Linq;

namespace GoodAI.Modules.Versioning
{
    public class MyConversion : MyBaseConversion
    {
        public override int CurrentVersion { get { return 7; } }


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
            result = result.Replace("yaxlib:realtype=\"GoodAI.Modules.Testing.MyRNGTask", "yaxlib:realtype=\"GoodAI.Modules.Common.MyRNGTask");             

            return result;
        }

        /// <summary>
        /// Convert LSTM delta tasks
        /// Author: KK
        /// </summary>
        public static string Convert4To5(string xml)
        {
            string result = xml;

            result = result.Replace("<Task Enabled=\"True\" PropertyName=\"deltaTask\" yaxlib:realtype=\"GoodAI.Modules.LSTM.Tasks.MyLSTMDeltaTask\" />", "");
            result = result.Replace("yaxlib:realtype=\"GoodAI.Modules.LSTM.Tasks.MyLSTMDummyDeltaTask", "yaxlib:realtype=\"GoodAI.Modules.LSTM.Tasks.MyLSTMDeltaTask");

            return result;
        }

        /// <summary>
        /// Convert LSTM activation function property names
        /// Author: KK
        /// </summary>
        public static string Convert5To6(string xml)
        {
            XDocument document = XDocument.Parse(xml);

            if (document.Root == null)
                return xml;

            foreach (var lstm in document.Root.Descendants("MyLSTMLayer"))
            {
                var activationFunction = lstm.Descendants("ActivationFunction").First();
                lstm.Add(new XElement("InputActivationFunction", activationFunction.Value));
                lstm.Add(new XElement("GateActivationFunction", "SIGMOID"));
            }

            return document.ToString();
        }

        /// <summary>
        /// Convert RandomNode task to multiple mutually exclusive tasks
        /// Author: MV
        /// </summary>
        public static string Convert6To7(string xml)
        {
            XDocument document = XDocument.Parse(xml);

            foreach (var node in document.Root.Descendants("MyRandomNode"))
            {
                XElement task = node.Descendants("Task").First();

                // Move period parameters from task to node
                string randomPeriod = task.Descendants("RandomPeriod").First().Value;
                string randomPeriodMin = task.Descendants("RandomPeriodMin").First().Value;
                string randomPeriodMax = task.Descendants("RandomPeriodMax").First().Value;
                string period = task.Descendants("Period").First().Value;

                node.Add(new XElement("RandomPeriod", randomPeriod));
                node.Add(new XElement("Period", period));
                node.Add(new XElement("RandomPeriodMin", randomPeriodMin));
                node.Add(new XElement("RandomPeriodMax", randomPeriodMax));

                XNamespace yaxlib = "http://www.sinairv.com/yaxlib/";
                XName realtype = yaxlib + "realtype";

                // Enable the correct task and transfer all task parameters
                string distribution = task.Descendants("Distribution").First().Value;

                XElement uniformTask = new XElement("Task");
                if (distribution == "Uniform")
                    uniformTask.SetAttributeValue("Enabled", "True");
                else
                    uniformTask.SetAttributeValue("Enabled", "False");
                uniformTask.SetAttributeValue("PropertyName", "UniformRNG");
                uniformTask.SetAttributeValue(realtype, "GoodAI.Modules.Common.MyUniformRNGTask");

                string uniformMin = task.Descendants("MinValue").First().Value;
                string uniformMax = task.Descendants("MaxValue").First().Value;
                uniformTask.Add(new XElement("MinValue", uniformMin));
                uniformTask.Add(new XElement("MaxValue", uniformMax));

                XElement normalTask = new XElement("Task");
                if (distribution == "Normal")
                    normalTask.SetAttributeValue("Enabled", "True");
                else
                    normalTask.SetAttributeValue("Enabled", "False");
                normalTask.SetAttributeValue("PropertyName", "NormalRNG");
                normalTask.SetAttributeValue(realtype, "GoodAI.Modules.Common.MyNormalRNGTask");

                string normalMean = task.Descendants("Mean").First().Value;
                string normalDev = task.Descendants("StdDev").First().Value;
                normalTask.Add(new XElement("Mean", normalMean));
                normalTask.Add(new XElement("StdDev", normalDev));

                XElement constantTask = new XElement("Task");
                if (distribution == "Constant")
                    constantTask.SetAttributeValue("Enabled", "True");
                else
                    constantTask.SetAttributeValue("Enabled", "False");
                constantTask.SetAttributeValue("PropertyName", "ConstantRNG");
                constantTask.SetAttributeValue(realtype, "GoodAI.Modules.Common.MyConstantRNGTask");

                string constantConstant = task.Descendants("Constant").First().Value;
                constantTask.Add(new XElement("Constant", constantConstant));

                XElement combinationTask = new XElement("Task");
                if (distribution == "Combination")
                    combinationTask.SetAttributeValue("Enabled", "True");
                else
                    combinationTask.SetAttributeValue("Enabled", "False");
                combinationTask.SetAttributeValue("PropertyName", "CombinationRNG");
                combinationTask.SetAttributeValue(realtype, "GoodAI.Modules.Common.MyCombinationRNGTask");

                string combinationMin = task.Descendants("Min").First().Value;
                string combinationMax = task.Descendants("Max").First().Value;
                string combinationUnique = task.Descendants("Unique").First().Value;
                combinationTask.Add(new XElement("Min", combinationMin));
                combinationTask.Add(new XElement("Max", combinationMax));
                combinationTask.Add(new XElement("Unique", combinationUnique));

                task.AddBeforeSelf(uniformTask);
                task.AddBeforeSelf(normalTask);
                task.AddBeforeSelf(constantTask);
                task.AddBeforeSelf(combinationTask);

                // Remove old task
                task.Remove();
            }

            return document.ToString();
        }
    }
}
