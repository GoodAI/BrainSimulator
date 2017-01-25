using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using System.IO;
using YAXLib;

namespace MNIST
{
    public abstract class ImageWorld : MyWorld
    {
        #region Abstract Properties
        protected abstract TensorDimensions InputDims { get; }
        protected abstract int NumberOfClasses { get; }
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
        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = false)]
        [Description("Binarize Input images with threshold = 0.5")]
        public bool Binarize { get; set; }

        [MyBrowsable, Category("Random"), DisplayName("Seed")]
        [YAXSerializableField(DefaultValue = 0)]
        [Description("The initial state of randomness generator, 0 = use random seed")]
        public int RandomSeed { get; set; }

        [MyBrowsable, Category("Random"), DisplayName("Example Order")]
        [YAXSerializableField(DefaultValue = ExampleOrderOption.Shuffle)]
        [Description("The order in which examples are presented")]
        public ExampleOrderOption ExampleOrder { get; set; }

        [MyBrowsable, Category("Target"), DisplayName("One-hot encoding")]
        [YAXSerializableField(DefaultValue = false)]
        [Description("Present target in the one-hot encoding instead of a class number")]
        public bool OneHot { get; set; }

        #endregion

        public override void UpdateMemoryBlocks()
        {
            Input.Dims = InputDims;

            if (OneHot)
            {
                Target.Dims = new TensorDimensions(NumberOfClasses);
                Target.MinValueHint = 0;
                Target.MaxValueHint = 1;
            }
            else
            {
                Target.Dims = new TensorDimensions(1);
                Target.MinValueHint = 0;
                Target.MaxValueHint = NumberOfClasses - 1;
            }

            //because values are normalized
            Input.MinValueHint = 0;
            Input.MaxValueHint = 1;
        }

        public void ValidateWorldSources(MyValidator validator, string[] paths, string baseDir, string datasetName, string url)
        {
            bool fail = false;
            foreach (string path in paths)
            {
                if (!File.Exists(path))
                {
                    validator.AddError(this, string.Format("Missing {0} dataset file \"{1}\". Check log (INFO level) for further information.", datasetName, Path.GetFileName(path)));
                    fail = true;
                }
            }

            if (fail)
            {
                MyLog.INFO.WriteLine("In order to use the {0} dataset, please visit:", datasetName);
                MyLog.INFO.WriteLine(url);
                MyLog.INFO.WriteLine("Then place the extracted files into:");
                MyLog.INFO.WriteLine(baseDir);
            }
        }
    }

    public abstract class SendDataTask : MyTask<ImageWorld>
    {
        private DatasetManager m_dataset;
        private ClassOrderOption m_classOrderOption;
        private bool m_useClassFilter;
        private string m_classFilter;
        private int m_nExamplesPerClass;

        [MyBrowsable, Category("Class Filter"), DisplayName("Enabled")]
        [YAXSerializableField(DefaultValue = false)]
        [Description("Filter classes")]
        public bool UseClassFilter
        {
            get { return m_useClassFilter; }
            set
            {
                m_useClassFilter = value;
                m_dataset.UseClassFilter(value);
            }
        }

        [MyBrowsable, Category("Class Filter"), DisplayName("Filter")]
        [YAXSerializableField(DefaultValue = "1,3,5")]
        [Description("Choose examples to be sent by the class number, e.g. '1,3,5'.")]
        public string ClassFilter
        {
            get { return m_classFilter;  }
            set
            {
                m_classFilter = value;
                m_dataset.SetClassFilter(value);
            }
        }

        [MyBrowsable, Category("Class Settings"), DisplayName("Examples per class")]
        [YAXSerializableField(DefaultValue = 5000)]
        [Description("Limit numer of examples per class")]
        public int ExamplesPerClass
        {
            get { return m_nExamplesPerClass; }
            set
            {
                m_nExamplesPerClass = m_dataset.SetExampleLimit(value);
            }
        }

        [MyBrowsable, Category("Class Settings"), DisplayName("Class Order")]
        [YAXSerializableField(DefaultValue = ClassOrderOption.Random)]
        [Description("The order of class from which examples are chosen")]
        public ClassOrderOption ClassOrder
        {
            get { return m_classOrderOption; }
            set
            {
                m_classOrderOption = value;
                m_dataset.ClassOrder = value;
            }
        }

        [MyBrowsable, Category("Params"), DisplayName("Exposition Time")]
        [YAXSerializableField(DefaultValue = 50)]
        [Description("How many time steps is each sample shown.")]
        public int ExpositionTime { get; set; }

        [MyBrowsable, Category("Params"), DisplayName("Exposition Time Offset")]
        [YAXSerializableField(DefaultValue = 0)]
        [Description("For how many time steps should blank be presented before real examples start to appear")]
        public int ExpositionTimeOffset { get; set; }


        public SendDataTask(AbstractDatasetReaderFactory datasetReaderFactory)
        {
            m_dataset = new DatasetManager(datasetReaderFactory);
        }

        public override void Init(int nGPU)
        {
            m_dataset.Init(Owner.RandomSeed, Owner.ExampleOrder);
            m_dataset.ClassOrder = ClassOrder;
            m_dataset.UseClassFilter(UseClassFilter);
            m_dataset.SetClassFilter(ClassFilter);
            m_dataset.SetExampleLimit(ExamplesPerClass);
        }

        public override void Execute()
        {
            if ((SimulationStep + ExpositionTimeOffset) % ExpositionTime == 0)
            {
                IExample ex = m_dataset.GetNext();

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
