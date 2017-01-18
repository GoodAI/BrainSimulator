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

        [MyBrowsable, Category("Random"), DisplayName("Seed"), Description("Used to set initial state randomness generator. 0 = use random seed")]
        [YAXSerializableField(DefaultValue = 0)]
        public int RandomSeed { get; set; }

        [MyBrowsable, Category("Example Order"), DisplayName("Mode"), Description("")] //TODO: fill in description
        [YAXSerializableField(DefaultValue = ExampleOrderOption.Shuffle)]
        public ExampleOrderOption ExampleOrder { get; set; }

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

    public abstract class SendDataTask : MyTask<ImageWorld>
    {
        protected DatasetManager _dataset;
        protected ClassOrderOption _classOrderOption;
        protected bool _useClassFilter;
        protected string _classFilter;


        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 50),
         Description("How many time steps is each sample shown.")]
        public int ExpositionTime { get; set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 0),
         Description("For how many time steps should blank be presented before real examples start to appear")]
        public int ExpositionTimeOffset { get; set; }

        private void UpdateManagerClassSettings()
        {
            if (_dataset != null)
            {
                if (UseClassFilter)
                {
                    _dataset.SetClassOrder(ClassOrder, ClassFilter);
                }
                else
                {
                    _dataset.SetClassOrder(ClassOrder);
                }
            }
        }

        [MyBrowsable, Category("Class order"), DisplayName("Mode")]
        [YAXSerializableField(DefaultValue = ClassOrderOption.Random),
         Description("")]
        public ClassOrderOption ClassOrder
        {
            get { return _classOrderOption; }
            set
            {
                _classOrderOption = value;
                UpdateManagerClassSettings();
            }
        }


        [MyBrowsable, Category("Class filter"), DisplayName("Enabled")]
        [YAXSerializableField(DefaultValue = false),
         Description("Filter classes")]
        public bool UseClassFilter
        {
            get { return _useClassFilter; }

            set
            {
                _useClassFilter = value;
                UpdateManagerClassSettings();
            }
        }

        [MyBrowsable, Category("Class filter"), DisplayName("Filter")]
        [YAXSerializableField(DefaultValue = "1,3,5"),
         Description("Choose examples to be sent by the class number, e.g. '1,3,5'.")]
        public string ClassFilter
        {
            get { return _classFilter;  }
            set
            {
                _classFilter = value;
                UpdateManagerClassSettings();
            }
        }

        protected abstract DatasetReaderFactory CreateFactory(string basePath);

        public override void Init(int nGPU)
        {
            string basePath = MyResources.GetMyAssemblyPath() + @"\res\";
            DatasetReaderFactory factory = CreateFactory(basePath);
            _dataset = new DatasetManager(factory, Owner.ExampleOrder, Owner.RandomSeed);
            UpdateManagerClassSettings();
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
