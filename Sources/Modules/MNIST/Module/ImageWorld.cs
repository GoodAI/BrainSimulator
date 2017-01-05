using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using YAXLib;

namespace MNIST
{
    public abstract class ImageWorld : MyWorld
    {
        #region Abstract Properties
        protected abstract TensorDimensions _inputDims { get; }
        protected abstract int _nClasses { get; }
        #endregion

        #region Memory Blocks
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Input
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> Target
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }
        #endregion

        #region World settings
        [MyBrowsable, Category("Input"), Description("Binarize Input images with threshold = 0.5")]
        [YAXSerializableField(DefaultValue = false)]
        public bool Binarize { get; set; }

        [MyBrowsable, Category("Target"), Description("Present target in one-hot encoding instead of class number")]
        [YAXSerializableField(DefaultValue = false)]
        public bool OneHot { get; set; }

        [MyBrowsable, Category("Random"), DisplayName("Seed"), Description("Used to shuffle the order in which examples get presented. 0 = use random seed")]
        [YAXSerializableField(DefaultValue = 0)]
        public int RandomSeed { get; set; }

        [MyBrowsable, Category("Random"), Description("Reshuffle dataset after each epoch.")]
        [YAXSerializableField(DefaultValue = true)]
        public bool Reshuffle { get; set; }

        #endregion

        public override void UpdateMemoryBlocks()
        {
            Input.Dims = _inputDims;

            if (OneHot)
            {
                Target.Dims = new TensorDimensions(_nClasses);
                Target.MinValueHint = 0;
                Target.MaxValueHint = 1;
            }
            else
            {
                Target.Dims = new TensorDimensions(1);
                Target.MinValueHint = 0;
                Target.MaxValueHint = _nClasses - 1;
            }

            //because values are normalized
            Input.MinValueHint = 0;
            Input.MaxValueHint = 1;
        }
    }

    [Description("Send Training Data"), MyTaskInfo(OneShot = false)]
    public abstract class SendDataTask : MyTask<ImageWorld>
    {
        protected DatasetManager _dataset;


        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 50),
         Description("How many time steps is each sample shown.")]
        public int ExpositionTime { get; set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 0),
         Description("For how many time steps should blank be presented before real examples start to appear")]
        public int ExpositionTimeOffset { get; set; }

        //[MyBrowsable, Category("Params"), DisplayName("Send numbers")]
        //[YAXSerializableField(DefaultValue = "All"),
        // Description("Choose data to be sent by class number, use 'All' or e.g. '1,3,5'")]
        //public string SendNumbers { get; set; }

        //[MyBrowsable, Category("Params"), DisplayName("Sequence ordered"),
        //Description("Send data ordered according to their labels (that is: 0,1,2,3,4..)?")]
        //[YAXSerializableField(DefaultValue = false)]
        //public bool SequenceOrdered { get; set; }


        protected abstract DatasetReaderFactory CreateFactory(string basePath);

        public override void Init(int nGPU)
        {
            string basePath = MyResources.GetMyAssemblyPath() + @"\res\";
            DatasetReaderFactory factory = CreateFactory(basePath);
            _dataset = new DatasetManager(factory, Owner.RandomSeed, Owner.Reshuffle);
        }

        public override void Execute()
        {
            if ((SimulationStep + ExpositionTimeOffset) % ExpositionTime == 0)
            {
                IExample ex = _dataset.GetNext();

                if (Owner.Binarize)
                {
                    for (int i = 0; i < ex.Input.Length; i++)
                    {
                        Owner.Input.Host[i] = ex.Input[i] >= 0.5 ? 1 : 0;
                    }
                }
                else
                {
                    Array.Copy(ex.Input, Owner.Input.Host, ex.Input.Length);
                }

                if (Owner.OneHot)
                {
                    Array.Clear(Owner.Target.Host, 0, 10);
                    Owner.Target.Host[ex.Target] = 1;
                }
                else
                {
                    Owner.Target.Host[0] = ex.Target;
                }

                Owner.Input.SafeCopyToDevice();
                Owner.Target.SafeCopyToDevice();
            }
        }
    }
}
