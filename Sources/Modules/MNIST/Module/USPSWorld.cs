
using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using System.ComponentModel;
using System;

namespace MNIST
{
    /// <author>GoodAI</author>
    /// <meta>mn</meta>
    /// <status>Work in progress</status>
    /// <summary>USPS dataset world</summary>
    /// <description>The world provides images from USPS dataset just like MNIST world does for MNIST dataset.</description>
    public class USPSWorld : ImageWorld
    {
        [MyTaskGroup("SendData")]
        public SendUSPSTrainDataTask SendTrainUSPSData { get; protected set; }
        [MyTaskGroup("SendData")]
        public SendUSPSTestDataTask SendTestUSPSData { get; protected set; }

        protected override TensorDimensions InputDims =>
            new TensorDimensions(USPSDatasetReader.ImageRows, USPSDatasetReader.ImageColumns, USPSDatasetReader.ImageChannels);
        protected override int NumberOfClasses => USPSDatasetReader.NumberOfClasses;
    }

    [Description("Send Train Data"), MyTaskInfo(OneShot = false)]
    public class SendUSPSTrainDataTask : SendDataTask
    {
        public SendUSPSTrainDataTask() :
            base(new USPSDatasetReaderFactory(USPSDatasetReader.DefaultTrainPath))
        {
        }
    }

    [Description("Send Test Data"), MyTaskInfo(OneShot = false)]
    public class SendUSPSTestDataTask : SendDataTask
    {
        public SendUSPSTestDataTask() :
            base(new USPSDatasetReaderFactory(USPSDatasetReader.DefaultTestPath))
        {
        }
    }
}
