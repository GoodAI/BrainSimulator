using GoodAI.Core.Configuration;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Linq;
using System;

namespace GoodAI.Modules.Versioning
{
    public class MyConversion : MyBaseConversion
    {
        public override int CurrentVersion { get { return 9; } }


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
        /// Convert version 6->7
        /// Author: Vision 
        /// </summary>
        public static string Convert6To7(string xml)
        {
            XDocument document = XDocument.Parse(xml);

            if (document.Root == null)
                return xml;

            // Rewire connections to layers inheriting from MyAbstractLayer (there's an extra input, "CanLearn")
            // Add tasks for neural network group
            foreach (var neuralNetworkGroup in document.Root.Descendants("Group"))
            {
                var groupTypeAttribute = neuralNetworkGroup.Attribute(GetRealTypeAttributeName());
                if (groupTypeAttribute != null && groupTypeAttribute.Value.EndsWith(".MyNeuralNetworkGroup"))
                {
                    Add6To7GroupTasks(neuralNetworkGroup);
//                    RewireLayerInputs(neuralNetworkGroup, document.Root.Descendants("Connection"));
                }
            }

            // Rename CrossEntropy field to CrossEntropyLoss
            foreach (var task in document.Root.Descendants("Task")) // go through all tasks because CrossEntropy is in multiple classes
            {
                var realTypeAttribute = task.Attribute(GetRealTypeAttributeName());
                if (realTypeAttribute != null && realTypeAttribute.Value.EndsWith(".MyCrossEntropyLossTask"))
                    task.Attribute("PropertyName").SetValue("CrossEntropyLoss");
            }

            // Move "Type" param from task to node
            foreach (var goniometricNode in document.Root.Descendants("GoniometricFunction"))
            {
                XElement gonioParams = goniometricNode.Descendants("Params").First();   // GoniometricFunction does not have "Params" so this yields task's "Params" field
                goniometricNode.Add(gonioParams);   // Type parameter in node is in "Params" section as well so this can be done
                gonioParams.Remove();   // removes only "Params" element in task; the new one in Node is intact
            }
            
            return document.ToString();
        }

        private static void RewireLayerInputs(XElement neuralNetworkGroup, IEnumerable<XElement> allConnections)
        {
            const int INDEX_OF_CAN_LEARN = 1;

            var children = neuralNetworkGroup.Element("Children");
            foreach (var layer in children.Elements())
            {
                string layerTypeName = layer.Attribute(GetRealTypeAttributeName()).Value;
                Type layerType = Type.GetType(layerTypeName);
                if (layerType != null)
                {
                    Type abstractLayerType = Type.GetType("GoodAI.Modules.NeuralNetwork.Layers.MyAbstractLayer");

                    if (layerType.IsSubclassOf(abstractLayerType))
                    {
                        string nodeID = layer.Attribute("Id").Value;
                        foreach (var connection in allConnections)
                        {
                            if (connection.Attribute("To").Value == nodeID)
                            {
                                var toIndexAttribute = connection.Attribute("ToIndex");
                                int toIndex = Convert.ToInt32(toIndexAttribute.Value);
                                if (toIndex >= INDEX_OF_CAN_LEARN)
                                {
                                    toIndexAttribute.SetValue(toIndex + 1);
                                }
                            }
                        }
                    }
                }
            }
        }

        private static void Add6To7GroupTasks(XElement neuralNetworkGroup)
        {
            var tasks = neuralNetworkGroup.Element("Tasks");

            var incrementTask = new XElement("Task");
            incrementTask.Add(new XAttribute("Enabled", "True"));
            incrementTask.Add(new XAttribute("PropertyName", "IncrementTimeStep"));
            incrementTask.Add(new XAttribute(GetRealTypeAttributeName(), "GoodAI.Modules.NeuralNetwork.Group.MyIncrementTimeStepTask"));
            tasks.Add(incrementTask);
            var decrementTask = new XElement("Task");
            decrementTask.Add(new XAttribute("Enabled", "True"));
            decrementTask.Add(new XAttribute("PropertyName", "DecrementTimeStep"));
            decrementTask.Add(new XAttribute(GetRealTypeAttributeName(), "GoodAI.Modules.NeuralNetwork.Group.MyDecrementTimeStepTask"));
            tasks.Add(decrementTask);
        }

        /// <summary>
        /// split focuser and unfocus :)
        /// </summary>
        /// <param name="xml"></param>
        /// <returns></returns>
        public static string Convert7To8(string xml)
        {
           XDocument document = XDocument.Parse(xml);

            if (document.Root == null)
                return xml;

            List<XElement> toRemove = new List<XElement>();

            foreach (var mapper in document.Root.Descendants("MyFocuser"))
            {
                {
                    foreach (var task in mapper.Descendants("Task"))
                    {
                        if (task.Attribute(GetRealTypeAttributeName()).Value == "GoodAI.Modules.Retina.MyFocuser+MyUnfocusTask")
                        {
                            toRemove.Add(task);
                        }

                    }
                }
            }

            foreach (var xElement in toRemove)
            {
                xElement.Remove();
            }
            return document.ToString();
        }

        /// <summary>
        /// QLearning tasks refactoring
        /// </summary>
        /// <param name="xml"></param>
        /// <returns></returns>
        public static string Convert8To9(string xml)
        {
            XDocument document = XDocument.Parse(xml);

            if (document.Root == null)
                return xml;

            List<XElement> toRemove = new List<XElement>();

            foreach (var task in document.Root.Descendants("Task"))
            {
                if (task.Attribute(GetRealTypeAttributeName()).Value == "GoodAI.Modules.NeuralNetwork.Tasks.MyRestoreValuesTask" ||
                    task.Attribute(GetRealTypeAttributeName()).Value == "GoodAI.Modules.NeuralNetwork.Tasks.MySaveActionTask")
                {
                    toRemove.Add(task);
                }
            }

            foreach (var xElement in toRemove)
            {
                xElement.Remove();
            }

            return document.ToString();
        }


        private static XName GetRealTypeAttributeName() 
        {
            XNamespace yaxlib = "http://www.sinairv.com/yaxlib/";
            XName realtype = yaxlib + "realtype";
            return realtype;
        }
    }
}
