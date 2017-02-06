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
    /// <description>
    /// The world provides RGB images from the CIFAR-10 dataset.<br/>
    /// There is 50000 training images and 10000 test images split equally within 10 classes.<br/>
    /// The 10 classes are (in this order of increasing class number): airplane, automobile, bird, cat, deer, dog, frog horse, ship and truck.
    /// </description>
    public class CIFAR10World : ImageWorld
    {
        [MyTaskGroup("SendData")]
        public SendCIFAR10TrainDataTask SendTrainCIFAR10Data { get; protected set; }
        [MyTaskGroup("SendData")]
        public SendCIFAR10TestDataTask SendTestCIFAR10Data { get; protected set; }

        protected override TensorDimensions BitmapDims =>
            new TensorDimensions(CIFAR10DatasetReader.ImageRows, CIFAR10DatasetReader.ImageColumns, CIFAR10DatasetReader.ImageChannels);
        protected override int NumberOfClasses => CIFAR10DatasetReader.NumberOfClasses;

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (!WorldSourcesExist(CIFAR10DatasetReader.DefaultNeededPaths, validator))
            {
                MyLog.INFO.WriteLine("In order to use the CIFAR-10 dataset, please visit:");
                MyLog.INFO.WriteLine("https://www.cs.toronto.edu/~kriz/cifar.html");
                MyLog.INFO.WriteLine("And download the binary version and extract the files into:");
                MyLog.INFO.WriteLine(CIFAR10DatasetReader.BaseDir);
            }
        }
    }

    /// <summary>Sends data from the train part of the CIFAR-10 dataset</summary>
    [Description("Send Train Data"), MyTaskInfo(OneShot = false)]
    public class SendCIFAR10TrainDataTask : SendDataTask
    {
        public SendCIFAR10TrainDataTask() :
            base(new CIFAR10DatasetReaderFactory(CIFAR10DatasetReader.DefaultTrainPaths))
        {
        }
    }

    /// <summary>Sends data from the test part of the CIFAR-10 dataset</summary>
    [Description("Send Test Data"), MyTaskInfo(OneShot = false)]
    public class SendCIFAR10TestDataTask : SendDataTask
    {
        public SendCIFAR10TestDataTask() :
            base(new CIFAR10DatasetReaderFactory(CIFAR10DatasetReader.DefaultTestPaths))
        {
        }
    }
}
