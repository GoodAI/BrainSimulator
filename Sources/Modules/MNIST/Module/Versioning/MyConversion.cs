using GoodAI.Core.Configuration;

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
            return xml;
        }
    }
}
