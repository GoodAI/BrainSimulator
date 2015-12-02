using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Signals;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using System.Linq;
using YAXLib;

namespace MNIST
{
    /// <author>GoodAI</author>
    /// <meta>mv</meta>
    /// <status>Working</status>
    /// <summary>Provides MNIST dataset images</summary>
    /// <description></description>
    public class MyMNISTWorld : MyWorld
    {
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Bitmap
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> Label
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyBrowsable, Category("Output"), Description("If set to true, output is binary vector of size 10")]
        [YAXSerializableField(DefaultValue = false)]
        public bool Binary { get; set; } // If set to true, output is binary vector of size 10

        [MyBrowsable, Category("Output"), Description("If set to true, pixels will have binary values 0/1 determined with threshold of 0.5.\nThe values have to be recomputed for every image, so it may slow down your simulation.")]
        [YAXSerializableField(DefaultValue = false)]
        public bool BinaryPixels { get; set; }

        public MyMNISTManager MNISTManager;

        public MyEOFSignal EOFSignal { get; private set; }
        public MyInitMNISTTask InitMNIST { get; private set; }
        public MySendMNISTTask SendMNISTData { get; private set; }

        public int ImagesCnt
        {
            get { return InitMNIST.ImagesCnt; }
        }

        public override void UpdateMemoryBlocks()
        {
            //MNIST images are 28 * 28 pixels
            Bitmap.Count = 28 * 28; // 784
            Bitmap.Dims = new TensorDimensions(28, 28);
            if (Binary)
            {
                Label.Count = 10;
                Label.Dims = new TensorDimensions(10);
                Label.MaxValueHint = Label.MinValueHint = 1;
            }
            else
            {
                Label.Count = 1;
                Label.MinValueHint = 0;
                Label.MaxValueHint = 9;
            }

            //because values are normalized
            Bitmap.MinValueHint = 0;
            Bitmap.MaxValueHint = 1;
        }
    }

    /// <summary>
    /// Initializes the MNIST world node. Subset size and repeating strategy can be set here.
    /// </summary>
    [Description("Init MNIST World"), MyTaskInfo(OneShot = true)]
    public class MyInitMNISTTask : MyTask<MyMNISTWorld>
    {
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = MNISTLastImageMethod.ResetToStart)]
        public MNISTLastImageMethod AfterLastImage { get; set; }

        private int m_imgsCnt = 2000;
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 2000)]
        public int ImagesCnt
        {
            get
            {
                return m_imgsCnt;
            }
            set
            {
                m_imgsCnt = value;
                if (Owner != null && Owner.MNISTManager != null)
                {
                    Owner.MNISTManager.m_imagesDemand = value;
                }
            }
        }

        public override void Init(int nGPU)
        {
            Owner.MNISTManager = new MyMNISTManager(MyResources.GetMyAssemblyPath() + @"\res\", ImagesCnt, false, AfterLastImage);
        }

        public override void Execute()
        {
        }
    }

    /// <summary>
    /// Sends MNIST image patch to the world output.
    /// <ul>
    /// <li><b>Exposition time</b> - For how many simulation steps each number is presented on the output</li>
    /// <li><b>Send Numbers</b> - All or enumeration of numbers requested on the output</li>
    /// <li><b>Random Order</b> - Sends patches in random order regardless to order in the dataset file</li>
    /// <li><b>Sequence Ordered</b> - Sends pathes in order defined in <b>Send Numbers</b> property</li>
    /// </ul>
    /// </summary>
    [Description("Send MNIST Data"), MyTaskInfo(OneShot = false)]
    public class MySendMNISTTask : MyTask<MyMNISTWorld>
    {
        private int[] m_numsToSend;
        private string m_send;

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 50)]
        public int ExpositionTime { get; set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 0)]
        public int ExpositionTimeOffset { get; set; }

        [MyBrowsable, Category("Params"), DisplayName("Send numbers")]
        [YAXSerializableField(DefaultValue = "All")]
        public string SendNumbers
        {
            get { return m_send; }
            set
            {
                m_send = value;
                if (SendNumbers.Equals("All") || SendNumbers.Equals("all"))
                {
                    m_numsToSend = Enumerable.Range(0, 10).ToArray();
                }
                else
                {
                    m_numsToSend = Array.ConvertAll(SendNumbers.Split(','), int.Parse);
                }
                if (Owner != null && Owner.MNISTManager != null)
                {
                    Owner.MNISTManager.m_sequenceIterator = 0;
                }
            }
        }

        [MyBrowsable, Category("Params"), DisplayName("Sequence ordered")]
        [YAXSerializableField(DefaultValue = false)]
        public bool SequenceOrdered { get; set; }


        [MyBrowsable, Category("Params"), DisplayName("Random order")]
        [YAXSerializableField(DefaultValue = false)]
        public bool RandomEnumerate { get; set; }

        public override void Init(int nGPU)
        {
        }

        public override void Execute()
        {
            Owner.EOFSignal.Drop();

            Owner.MNISTManager.RandomEnumerate = RandomEnumerate;
            Owner.MNISTManager.m_definedOrder = SequenceOrdered;

            //if ((SimulationStep <= ExpositionTime * Owner.ImagesCnt) && (SimulationStep % ExpositionTime == 0))
            if ((SimulationStep + ExpositionTimeOffset) % ExpositionTime == 0)
            {
                MyMNISTImage im = Owner.MNISTManager.GetNextImage(m_numsToSend);

                if (Owner.BinaryPixels)
                    im.ToBinary();

                Array.Copy(im.Data1D, Owner.Bitmap.Host, 784);   //28 * 28
                Owner.Bitmap.SafeCopyToDevice();

                if (Owner.Binary)
                {
                    Array.Clear(Owner.Label.Host, 0, 10);
                    Owner.Label.Host[im.Label] = 1;
                }
                else
                {
                    Owner.Label.Host[0] = im.Label;
                }
                Owner.Label.SafeCopyToDevice();
                Owner.EOFSignal.Raise();
            }
        }
    }
}