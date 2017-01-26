using GoodAI.Core.Configuration;
using System.Xml.Linq;
using System.Collections.Generic;

namespace MNIST.Versioning
{
    public class MyConversion : MyBaseConversion
    {
        public override int CurrentVersion => 3;

        /// <summary>
        /// Convert MNISTWorld brains for the training and testing tasks. 
        /// Enable training by default, so that the functionality is preserved.
        /// Author: jv
        /// </summary>
        public static string Convert1To2(string xml)
        {
            xml = xml.Replace("MNIST.MySendMNISTTask", "MNIST.MySendTrainingMNISTTask");
            xml = xml.Replace("SendMNISTData", "SendTrainingMNISTData");
            xml = xml.Replace("ImagesCnt", "TrainingExamplesPerDigit");

            XDocument document = XDocument.Parse(xml);
            if (document.Root == null)
                return xml;

            foreach (XElement element in document.Root.Descendants("TrainingExamplesPerDigit"))
            {
                int oldValue = int.Parse(element.Value) / 10;
                element.SetValue(oldValue.ToString());
            }

            return document.ToString();
        }

        public static string Convert2To3(string xml)
        {

            XDocument document = XDocument.Parse(xml);

            foreach (XElement worldNode in document.Root.Descendants("World"))
            {
                XAttribute worldAtt = worldNode.Attribute(GetRealTypeAttributeName());

                if (worldAtt.Value == "MNIST.MyMNISTWorld")
                {
                    worldAtt.Value = "MNIST.MNISTWorld";

                    // one-hot encoding conversion
                    #region
                    foreach (XElement node in worldNode.Descendants("Binary"))
                    {
                        node.Name = "OneHot";
                    }
                    #endregion

                    // image binarize conversion
                    #region
                    foreach (XElement node in worldNode.Descendants("BinaryPixels"))
                    {
                        node.Name = "Binarize";
                    }
                    #endregion

                    int? trainExamplesPerClass = null;
                    int? testExamplesPerClass = null;

                    // Get number of examples per class from Init task
                    #region 
                    foreach (XElement taskNode in worldNode.Descendants("Task"))
                    {
                        XAttribute taskAtt = taskNode.Attribute(GetRealTypeAttributeName());
                        if (taskAtt.Value == "MNIST.MyInitMNISTTask")
                        {
                            foreach (XElement node in taskNode.Descendants("TrainingExamplesPerDigit"))
                            {
                                trainExamplesPerClass = int.Parse(node.Value);
                            }

                            foreach (XElement node in taskNode.Descendants("TestExamplesPerDigit"))
                            {
                                testExamplesPerClass = int.Parse(node.Value);
                            }
                            taskNode.Remove();
                            break;
                        }
                    }
                    #endregion

                    foreach (XElement taskNode in worldNode.Descendants("Task"))
                    {
                        bool isTrain = false;
                        XAttribute taskAtt;

                        // task's attributes renaming + determine if current task is training task - set in isTrain
                        #region 
                        taskAtt = taskNode.Attribute(GetRealTypeAttributeName());
                        if (taskAtt.Value == "MNIST.MySendTrainingMNISTTask")
                        {
                            taskAtt.Value = "MNIST.SendMNISTTrainDataTask";
                            isTrain = true;
                        }
                        else if (taskAtt.Value == "MNIST.MySendTestMNISTTask")
                        {
                            taskAtt.Value = "MNIST.SendMNISTTestDataTask";
                        }

                        taskAtt = taskNode.Attribute("PropertyName");
                        if (taskAtt.Value == "SendTrainingMNISTData")
                        {
                            taskAtt.Value = "SendMNISTTrainData";
                        }
                        else if (taskAtt.Value == "SendTestMNISTData")
                        {
                            taskAtt.Value = "SendMNISTTestData";
                        }
                        #endregion

                        // class filter conversion
                        #region
                        bool useClassFilter = false;
                        foreach (XElement node in taskNode.Descendants("SendNumbers"))
                        {
                            if (node.Value == "All")
                            {
                                node.Remove();
                                useClassFilter = false;
                                break;
                            }
                            else
                            {
                                node.Name = "ClassFilter";
                                useClassFilter = true;
                                break;
                            }

                        }
                        taskNode.Add(new XElement("UseClassFilter", useClassFilter));
                        #endregion

                        // class order conversion
                        #region
                        foreach (XElement node in taskNode.Descendants("SequenceOrdered"))
                        {
                            node.Name = "ClassOrder";
                            if (node.Value == "True")
                            {
                                node.Value = ClassOrderOption.Increasing.ToString();
                            }
                            else
                            {
                                node.Value = ClassOrderOption.Random.ToString();
                            }
                        }
                        #endregion

                        // example order conversion (from the train task)
                        #region
                        foreach (XElement node in taskNode.Descendants("RandomEnumerate"))
                        {
                            ExampleOrderOption exampleOrder; 
                            if (node.Value == "True")
                            {
                                exampleOrder = ExampleOrderOption.Shuffle;
                            }
                            else
                            {
                                exampleOrder = ExampleOrderOption.NoShuffle;
                            }

                            if (isTrain)
                            {
                                worldNode.Add(new XElement("ExampleOrder", exampleOrder));
                            }

                            node.Remove();
                            break;
                        }
                        #endregion

                        // examples per class conversion (from previously read values from Init task)
                        #region
                        if (isTrain)
                        {
                            if (trainExamplesPerClass != null)
                            {
                                taskNode.Add(new XElement("ExamplesPerClass", trainExamplesPerClass));
                            }
                        }
                        else
                        {
                            if (trainExamplesPerClass != null)
                            {
                                taskNode.Add(new XElement("ExamplesPerClass", testExamplesPerClass));
                            }

                        }
                        #endregion
                    }
                }
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
