
using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using System;
using System.IO;
using System.ComponentModel;

namespace MNIST
{
    /// <author>GoodAI</author>
    /// <meta>mn</meta>
    /// <status>Work in progress</status>
    /// <summary>New version of MNIST World (2017)</summary>
    /// <description>
    /// This world provides grayscale images from the MNIST dataset of handwritten digits with the following distribution of images among classes:<br/>
    /// <table>
    /// <thead>
    /// <tr> <th></th> <th>0</th> <th>1</th> <th>2</th> <th>3</th> <th>4</th> <th>5</th> <th>6</th> <th>7</th> <th>8</th> <th>9</th> <th>Total</th> </tr>
    /// </thead>
    /// <tbody>
    /// <tr> <td>Train</td> <td>5923</td> <td>6742</td> <td>5958</td> <td>6131</td> <td>5842</td> <td>5421</td> <td>5918</td> <td>6265</td> <td>5851</td> <td>5949</td> <td>60000</td> </tr>
    /// <tr> <td>Test</td> <td>980</td> <td>1135</td> <td>1032</td> <td>1010</td> <td>982</td> <td>892</td> <td>958</td> <td>1028</td> <td>974</td> <td>1009</td> <td>10000</td> </tr>
    /// </tbody>
    /// </table>
    /// </description>

    public class MNISTWorld : ImageWorld
    {
        [MyTaskGroup("SendData")]
        public SendMNISTTrainDataTask SendMNISTTrainData { get; protected set; }
        [MyTaskGroup("SendData")]
        public SendMNISTTestDataTask SendMNISTTestData { get; protected set; }

        protected override TensorDimensions InputDims =>
            new TensorDimensions(MNISTDatasetReader.ImageRows, MNISTDatasetReader.ImageColumns, MNISTDatasetReader.ImageChannels);
        protected override int NumberOfClasses => MNISTDatasetReader.NumberOfClasses;

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (!WorldSourcesExist(MNISTDatasetReader.DefaultNeededPaths, validator))
            {
                MyLog.INFO.WriteLine("In order to use the MNIST dataset, please visit:");
                MyLog.INFO.WriteLine("http://yann.lecun.com/exdb/mnist/");
                MyLog.INFO.WriteLine("And download and extract the files into:");
                MyLog.INFO.WriteLine(MNISTDatasetReader.BaseDir);
            }
        }
    }

    /// <summary>Sends data from the train part of the MNIST dataset</summary>
    [Description("Send Train Data"), MyTaskInfo(OneShot = false)]
    public class SendMNISTTrainDataTask : SendDataTask
    {
        public SendMNISTTrainDataTask() :
            base(new MNISTDatasetReaderFactory(MNISTDatasetReader.DefaultTrainImagePath, MNISTDatasetReader.DefaultTrainLabelPath))
        {
        }
    }

    /// <summary>Sends data from the test part of the MNIST dataset</summary>
    [Description("Send Test Data"), MyTaskInfo(OneShot = false)]
    public class SendMNISTTestDataTask : SendDataTask
    {
        public SendMNISTTestDataTask() :
            base(new MNISTDatasetReaderFactory(MNISTDatasetReader.DefaultTestImagePath, MNISTDatasetReader.DefaultTestLabelPath))
        {
        }
    }
}
