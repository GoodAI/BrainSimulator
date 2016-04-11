using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Signals;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using System.Linq;
using System.Collections.Generic;
using YAXLib;

namespace MNIST
{
    /// <author>GoodAI</author>
    /// <meta>mv</meta>
    /// <status>Working</status>
    /// <summary>Provides MNIST dataset images.</summary>
    /// <description>There is 60000 (roughly 6000 for each class) and training and 10000 testing images.</description>
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

        // mutually exclusive tasks
        [MyTaskGroup("SendData")]
        public MySendTrainingMNISTTask SendTrainingMNISTData { get; protected set; }
        [MyTaskGroup("SendData")]
        public MySendTestMNISTTask SendTestMNISTData { get; protected set; }

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

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if ((InitMNIST.TrainingExamplesPerDigit < 1) || (InitMNIST.TrainingExamplesPerDigit > 7000) ||
                (InitMNIST.TestExamplesPerDigit < 1) || (InitMNIST.TestExamplesPerDigit > 7000))
            {
                validator.AddError(this, "The value of ExamplesPerDigit properties of the Init task have to be in the interval <1, 7000>");
            }
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

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 7000),
        DisplayName("Train Examples per Digit"),
        Description("Continue loading images from the dataset until there is required number of training samples for each digit.")]
        public int TrainingExamplesPerDigit
        {
            get
            {
                return m_trainingExamplesPerDigit;
            }
            set
            {
                m_trainingExamplesPerDigit = value;
            }
        }

        [MyBrowsable, Category("Params"), DisplayName("Test Examples per Digit"),
        Description("Continue loading images from the dataset until there is required number of testing samples for each digit.")]
        [YAXSerializableField(DefaultValue = 2000)]
        public int TestExamplesPerDigit
        {
            get
            {
                return m_testExamplesPerDigit;
            }
            set
            {
                m_testExamplesPerDigit = value;
            }
        }

        public override void Init(int nGPU)
        {
            Owner.MNISTManager = new MyMNISTManager(MyResources.GetMyAssemblyPath() + @"\res\",
                TrainingExamplesPerDigit, TestExamplesPerDigit, false, AfterLastImage);
        }

        public override void Execute()
        {
        }

        private int m_trainingExamplesPerDigit;
        private int m_testExamplesPerDigit;
    }

    // Parent class ment to be derived from MySendTrainingMNISTTask and MySendTestMNISTTask (below)
    // Its done in this manner so each set (training/test) can be set-up independently and its easilly switchable in the GUI
    public class MySendMNISTTask : MyTask<MyMNISTWorld>
    {
        protected MNIST.MNISTSetType m_setType;

        public int[] m_numsToSend { get; protected set; }
        private string m_send;

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 50),
         Description("How many time steps is each sample shown.")]
        public int ExpositionTime { get; set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 0)]
        public int ExpositionTimeOffset { get; set; }

        [MyBrowsable, Category("Params"), DisplayName("Send numbers")]
        [YAXSerializableField(DefaultValue = "All"),
         Description("Choose data to be sent by labels, use 'All' or e.g. '1,3,5'")]
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

        [MyBrowsable, Category("Params"), DisplayName("Sequence ordered"),
        Description("Send data ordered according to their labels (that is: 0,1,2,3,4..)?")]
        [YAXSerializableField(DefaultValue = false)]
        public bool SequenceOrdered { get; set; }


        [MyBrowsable, Category("Params"), DisplayName("Random order"),
        Description("Read data from the dataset in random order? Can be combined with Sequence Ordered parameter.")]
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
                MyMNISTImage im = Owner.MNISTManager.GetNextImage(m_numsToSend, m_setType);

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

    /// <summary>
    /// Sends MNIST TRAINING image patch to the world output.
    /// <ul>
    /// <li><b>Exposition time</b> - For how many simulation steps each number is presented on the output.</li>
    /// <li><b>Send Numbers</b> - Sends numbers from all classes ('All') or only from classes enumerated in the field (e.g. '1,2,3').</li>
    /// <li><b>Random Order</b> - Sends patches in random order regardless to order in the dataset file.</li>
    /// <li><b>Sequence Ordered</b> - Order the images by their class labels (i.e. sends sequence of images with labels '0,1,2,3,4...')?</li>
    /// </ul>
    /// </summary>
    [Description("Send Training Data"), MyTaskInfo(OneShot = false)]
    public class MySendTrainingMNISTTask : MySendMNISTTask
    {
        public MySendTrainingMNISTTask()
            : base()
        {
            m_setType = MNIST.MNISTSetType.Training;
        }
    }

    /// <summary>
    /// Sends MNIST TESTING image patch to the world output.
    /// <ul>
    /// <li><b>Exposition time</b> - For how many simulation steps each number is presented on the output.</li>
    /// <li><b>Send Numbers</b> - Sends numbers from all classes ('All') or only from classes enumerated in the field (e.g. '1,2,3').</li>
    /// <li><b>Random Order</b> - Sends patches in random order regardless to order in the dataset file.</li>
    /// <li><b>Sequence Ordered</b> - Order the images by their class labels (i.e. sends sequence of images with labels '0,1,2,3,4...')?</li>
    /// </ul>
    /// </summary>
    [Description("Send Test Data"), MyTaskInfo(OneShot = false)]
    public class MySendTestMNISTTask : MySendMNISTTask
    {
        public MySendTestMNISTTask()
            : base()
        {
            m_setType = MNIST.MNISTSetType.Test;
        }
    }

}