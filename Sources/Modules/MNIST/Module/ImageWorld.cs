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
        public MyMemoryBlock<float> Bitmap
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> Class
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

        [MyBrowsable, Category("Random"), DisplayName("Bitmap order")]
        [YAXSerializableField(DefaultValue = ExampleOrderOption.Shuffle)]
        [Description(@"The order in which bitmaps are presented. Bitmaps are always shuffled before start.
NoShuffle = the order of bitmaps within each class then remains fixed
Shuffle = once last bitmap of the requested class has been served, the bitmaps within this class get shuffled again
RandomSample = sample random bitmap from the requested class")]
        public ExampleOrderOption BitmapOrder { get; set; }

        [MyBrowsable, Category("Target"), DisplayName("One-hot encoding")]
        [YAXSerializableField(DefaultValue = false)]
        [Description("Present target in the one-hot encoding instead of a class number")]
        public bool OneHot { get; set; }

        #endregion

        public override void UpdateMemoryBlocks()
        {
            Bitmap.Dims = InputDims;

            if (OneHot)
            {
                Class.Dims = new TensorDimensions(NumberOfClasses);
                Class.MinValueHint = 0;
                Class.MaxValueHint = 1;
            }
            else
            {
                Class.Dims = new TensorDimensions(1);
                Class.MinValueHint = 0;
                Class.MaxValueHint = NumberOfClasses - 1;
            }

            //because values are normalized
            Bitmap.MinValueHint = 0;
            Bitmap.MaxValueHint = 1;
        }

        protected bool WorldSourcesExist(string[] paths, MyValidator validator)
        {
            bool exist = true;
            foreach (string path in paths)
            {
                if (!File.Exists(path))
                {
                    validator.AddError(this, string.Format("Missing dataset file \"{1}\". Check log (INFO level) for further information.", Path.GetFileName(path)));
                    exist = false;
                }
            }

            return exist;
        }
    }

    public abstract class SendDataTask : MyTask<ImageWorld>
    {
        private DatasetManager m_dataset;
        private ClassOrderOption m_classOrderOption;
        private bool m_useClassFilter;
        private string m_classFilter;
        private int m_nBitmapsPerClass;

        [MyBrowsable, Category("Class Filter"), DisplayName("Filter")]
        [YAXSerializableField(DefaultValue = "1,3,5")]
        [Description("Choose bitmaps to be sent by the class number, e.g. '1,3,5'.")]
        public string ClassFilter
        {
            get { return m_classFilter;  }
            set
            {
                m_classFilter = value;
                m_dataset.UseClassFilter(UseClassFilter, ConvertFilter(value));
            }
        }

        [MyBrowsable, Category("Class Filter"), DisplayName("Enabled")]
        [YAXSerializableField(DefaultValue = false)]
        [Description("Filter classes")]
        public bool UseClassFilter
        {
            get { return m_useClassFilter; }
            set
            {
                m_useClassFilter = value;
                m_dataset.UseClassFilter(value, ConvertFilter(ClassFilter));
            }
        }


        [MyBrowsable, Category("Class Settings"), DisplayName("Bitmaps per class")]
        [YAXSerializableField(DefaultValue = 5000)]
        [Description("Limit numer of bitmaps per class")]
        public int BitmapsPerClass
        {
            get { return m_nBitmapsPerClass; }
            set
            {
                m_nBitmapsPerClass = m_dataset.SetExampleLimit(value);
            }
        }

        [MyBrowsable, Category("Class Settings"), DisplayName("Class order")]
        [YAXSerializableField(DefaultValue = ClassOrderOption.Random)]
        [Description("The order of class from which bitmaps are chosen")]
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
        [Description("For how many time steps should blank be presented before bitmaps from dataset start to appear")]
        public int ExpositionTimeOffset { get; set; }

        private static int[] ConvertFilter(string filter)
        {
            string[] strClasses = filter.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            return Array.ConvertAll(strClasses, int.Parse);
        }

        public SendDataTask(AbstractDatasetReaderFactory datasetReaderFactory)
        {
            m_dataset = new DatasetManager(datasetReaderFactory);
        }

        public override void Init(int nGPU)
        {
            m_dataset.Init(Owner.BitmapOrder, Owner.RandomSeed);
            m_dataset.ClassOrder = ClassOrder;
            m_dataset.UseClassFilter(UseClassFilter, ConvertFilter(ClassFilter));
            m_nBitmapsPerClass = m_dataset.SetExampleLimit(BitmapsPerClass); // TODO: user has to select property first before it visually updates its value
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
                        Owner.Bitmap.Host[i] = ex.Input[i] >= 0.5 ? 1 : 0;
                    }
                }
                else
                {
                    Array.Copy(ex.Input, Owner.Bitmap.Host, ex.Input.Length);
                }

                if (Owner.OneHot)
                {
                    Array.Clear(Owner.Class.Host, 0, 10);
                    Owner.Class.Host[ex.Target] = 1;
                }
                else
                {
                    Owner.Class.Host[0] = ex.Target;
                }

                Owner.Bitmap.SafeCopyToDevice();
                Owner.Class.SafeCopyToDevice();
            }
        }
    }
}
