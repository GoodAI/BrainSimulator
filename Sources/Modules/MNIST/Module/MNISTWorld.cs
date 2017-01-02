
using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;

namespace MNIST
{
    /// <author>GoodAI</author>
    /// <meta>mn</meta>
    /// <status>Work in progress</status>
    /// <summary>New version of MNIST World (2017)</summary>
    /// <description>There is 60000 (roughly 6000 for each class) and training and 10000 testing images.</description>
    public class MNISTWorld : ImageWorld
    {
        [MyTaskGroup("SendData")]
        public SendMNISTTrainDataTask SendTrainMNISTData { get; protected set; }
        [MyTaskGroup("SendData")]
        public SendMNISTTestDataTask SendTestMNISTData { get; protected set; }

        public override void UpdateMemoryBlocks()
        {
            Input.Dims = new TensorDimensions(MNISTDatasetReader.ImageRows, MNISTDatasetReader.ImageColumns);

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
                Target.MaxValueHint = MNISTDatasetReader.NumberOfClasses - 1;
            }

            //because values are not normalized
            Input.MinValueHint = 0;
            Input.MaxValueHint = 255;
        }
    }

    [Description("Send Train Data"), MyTaskInfo(OneShot = false)]
    public class SendMNISTTrainDataTask : SendDataTask
    {
        public override void Init(int nGPU)
        {
            string basePath = MyResources.GetMyAssemblyPath() + @"\res\";
            DatasetReaderFactory factory = new MNISTDatasetReaderFactory(basePath, DatasetReaderFactoryType.Train);
            _dataset = new DatasetManager(factory);
        }
    }

    [Description("Send Test Data"), MyTaskInfo(OneShot = false)]
    public class SendMNISTTestDataTask : SendDataTask
    {
        public override void Init(int nGPU)
        {
            string basePath = MyResources.GetMyAssemblyPath() + @"\res\";
            DatasetReaderFactory factory = new MNISTDatasetReaderFactory(basePath, DatasetReaderFactoryType.Test);
            _dataset = new DatasetManager(factory);
        }
    }
}
