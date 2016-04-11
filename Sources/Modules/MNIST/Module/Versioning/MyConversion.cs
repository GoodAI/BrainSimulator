using GoodAI.Core.Configuration;
using System.Xml.Linq;
using System.Collections.Generic;

namespace MNIST.Versioning
{
    public class MyConversion : MyBaseConversion
    {
        public override int CurrentVersion
        {
            get { return 2; }
        }

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
    }
}
