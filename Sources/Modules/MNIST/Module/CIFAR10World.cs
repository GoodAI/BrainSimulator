using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using System.ComponentModel;
using System;
using System.IO;

namespace MNIST
{
    /// <author>GoodAI</author>
    /// <meta>mn</meta>
    /// <status>Work in progress</status>
    /// <summary>CIFAR-10 dataset world</summary>
    /// <description>The world provides images from the CIFAR-10 dataset just like MNIST world does for MNIST dataset.</description>
    public class CIFAR10World : ImageWorld
    {
        [MyTaskGroup("SendData")]
        public SendCIFAR10TrainDataTask SendTrainCIFAR10Data { get; protected set; }
        [MyTaskGroup("SendData")]
        public SendCIFAR10TestDataTask SendTestCIFAR10Data { get; protected set; }

        protected override TensorDimensions InputDims =>
            new TensorDimensions(CIFAR10DatasetReader.ImageRows, CIFAR10DatasetReader.ImageColumns, CIFAR10DatasetReader.ImageChannels);
        protected override int NumberOfClasses => CIFAR10DatasetReader.NumberOfClasses;

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            ValidateWorldSources(validator, CIFAR10DatasetReader.DefaultNeededPaths, CIFAR10DatasetReader.BaseDir, "CIFAR-10", "https://www.cs.toronto.edu/~kriz/cifar.html");
        }
    }

    [Description("Send Train Data"), MyTaskInfo(OneShot = false)]
    public class SendCIFAR10TrainDataTask : SendDataTask
    {
        public SendCIFAR10TrainDataTask() :
            base(new CIFAR10DatasetReaderFactory(CIFAR10DatasetReader.DefaultTrainPaths))
        {
        }
    }

    [Description("Send Test Data"), MyTaskInfo(OneShot = false)]
    public class SendCIFAR10TestDataTask : SendDataTask
    {
        public SendCIFAR10TestDataTask() :
            base(new CIFAR10DatasetReaderFactory(CIFAR10DatasetReader.DefaultTestPaths))
        {
        }
    }
}
