using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using System.IO;
using System.ComponentModel;

namespace MNIST
{
    /// <author>GoodAI</author>
    /// <meta>mn</meta>
    /// <status>Work in progress</status>
    /// <summary>USPS dataset world</summary>
    /// <description>
    /// This world provides grayscale images from the USPS dataset of handwritten digits with the following distribution of images among classes:<br/>
    /// <table>
    /// <thead>
    /// <tr> <th></th> <th>0</th> <th>1</th> <th>2</th> <th>3</th> <th>4</th> <th>5</th> <th>6</th> <th>7</th> <th>8</th> <th>9</th> <th>Total</th> </tr>
    /// </thead>
    /// <tbody>
    /// <tr> <td>Train</td> <td>1194</td> <td>1005</td> <td>731</td> <td>658</td> <td>652</td> <td>556</td> <td>664</td> <td>645</td> <td>542</td> <td>644</td> <td>7291</td> </tr>
    /// <tr> <td>Test</td> <td>359</td> <td>264</td> <td>198</td> <td>166</td> <td>200</td> <td>160</td> <td>170</td> <td>147</td> <td>166</td> <td>177</td> <td>2007</td> </tr>
    /// </tbody>
    /// </table>
    /// </description>
    public class USPSWorld : ImageWorld
    {
        [MyTaskGroup("SendData")]
        public SendUSPSTrainDataTask SendTrainUSPSData { get; protected set; }
        [MyTaskGroup("SendData")]
        public SendUSPSTestDataTask SendTestUSPSData { get; protected set; }

        protected override TensorDimensions InputDims =>
            new TensorDimensions(USPSDatasetReader.ImageRows, USPSDatasetReader.ImageColumns, USPSDatasetReader.ImageChannels);
        protected override int NumberOfClasses => USPSDatasetReader.NumberOfClasses;

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (!WorldSourcesExist(USPSDatasetReader.DefaultNeededPaths, validator))
            {
                MyLog.INFO.WriteLine("In order to use the USPS dataset, please visit:");
                MyLog.INFO.WriteLine("http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps");
                MyLog.INFO.WriteLine("And download both dataset files and extract them into:");
                MyLog.INFO.WriteLine(USPSDatasetReader.BaseDir);
            }
        }
    }

    /// <summary>Sends data from the train part of the USPS dataset</summary>
    [Description("Send Train Data"), MyTaskInfo(OneShot = false)]
    public class SendUSPSTrainDataTask : SendDataTask
    {
        public SendUSPSTrainDataTask() :
            base(new USPSDatasetReaderFactory(USPSDatasetReader.DefaultTrainPath))
        {
        }
    }

    /// <summary>Sends data from the test part of the USPS dataset</summary>
    [Description("Send Test Data"), MyTaskInfo(OneShot = false)]
    public class SendUSPSTestDataTask : SendDataTask
    {
        public SendUSPSTestDataTask() :
            base(new USPSDatasetReaderFactory(USPSDatasetReader.DefaultTestPath))
        {
        }
    }
}
