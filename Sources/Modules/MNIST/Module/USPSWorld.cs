
using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using System.ComponentModel;

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

        public override void UpdateMemoryBlocks()
        {
            Input.Dims = new TensorDimensions(USPSDatasetReader.ImageRows, USPSDatasetReader.ImageColumns);

            if (OneHot)
            {
                Target.Dims = new TensorDimensions(10);
                Target.MinValueHint = 0;
                Target.MaxValueHint = 1;
            }
            else
            {
                Target.Dims = new TensorDimensions(1);
                Target.MinValueHint = 0;
                Target.MaxValueHint = USPSDatasetReader.NumberOfClasses - 1;
            }

            //because values are not normalized
            Input.MinValueHint = -1;
            Input.MaxValueHint = 1;
        }
    }

    [Description("Send Train Data"), MyTaskInfo(OneShot = false)]
    public class SendUSPSTrainDataTask : SendDataTask
    {
        public override void Init(int nGPU)
        {
            string basePath = MyResources.GetMyAssemblyPath() + @"\res\";
            DatasetReaderFactory factory = new USPSDatasetReaderFactory(basePath, DatasetReaderFactoryType.Train);
            _dataset = new DatasetManager(factory);
        }
    }

    [Description("Send Test Data"), MyTaskInfo(OneShot = false)]
    public class SendUSPSTestDataTask : SendDataTask
    {
        public override void Init(int nGPU)
        {
            string basePath = MyResources.GetMyAssemblyPath() + @"\res\";
            DatasetReaderFactory factory = new USPSDatasetReaderFactory(basePath, DatasetReaderFactoryType.Test);
            _dataset = new DatasetManager(factory);
        }
    }
}
