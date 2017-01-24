
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
    /// <description>There is 60000 (roughly 6000 for each class) training and 10000 testing images.</description>
    public class MNISTWorld : ImageWorld
    {
        [MyTaskGroup("SendData")]
        public SendMNISTTrainDataTask SendTrainMNISTData { get; protected set; }
        [MyTaskGroup("SendData")]
        public SendMNISTTestDataTask SendTestMNISTData { get; protected set; }

        protected override TensorDimensions InputDims
        {
            get
            {
                return new TensorDimensions(MNISTDatasetReader.ImageRows, MNISTDatasetReader.ImageColumns, MNISTDatasetReader.ImageChannels);
            }
        }

        protected override int NumberOfClasses
        {
            get
            {
                return MNISTDatasetReader.NumberOfClasses;
            }
        }
    }

    [Description("Send Train Data"), MyTaskInfo(OneShot = false)]
    public class SendMNISTTrainDataTask : SendDataTask
    {
        protected override DatasetReaderFactory CreateFactory(string basePath)
        {
            return new MNISTDatasetReaderFactory(basePath, DatasetReaderFactoryType.Train);
        }
    }

    [Description("Send Test Data"), MyTaskInfo(OneShot = false)]
    public class SendMNISTTestDataTask : SendDataTask
    {
        protected override DatasetReaderFactory CreateFactory(string basePath)
        {
            return new MNISTDatasetReaderFactory(basePath, DatasetReaderFactoryType.Test);
        }
    }
}
