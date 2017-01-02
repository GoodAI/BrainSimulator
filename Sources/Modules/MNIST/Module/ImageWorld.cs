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
        //[MyBrowsable, Category("Input"), Description("Binarize Input images with threshold = 0.5")]
        //[YAXSerializableField(DefaultValue = false)]
        //public bool Binarize { get; set; }

        [MyBrowsable, Category("Target"), Description("Present target in one-hot encoding instead of class number")]
        [YAXSerializableField(DefaultValue = false)]
        public bool OneHot { get; set; }

        //[MyBrowsable, Category("Random"), Description("Used to shuffle the order in which examples get presented.\n0 = use random RandomSeed")]
        //[YAXSerializableField(DefaultValue = 0)]
        //public int RandomSeed { get; set; }
        #endregion
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


        //[MyBrowsable, Category("Params"), DisplayName("Random order"),
        //Description("Reshuffle dataset after each epoch.")]
        //[YAXSerializableField(DefaultValue = false)]
        //public bool Reshuffle { get; set; }

        public override void Execute()
        {
            if ((SimulationStep + ExpositionTimeOffset) % ExpositionTime == 0)
            {
                IExample ex = _dataset.GetNext();
                Array.Copy(ex.Input, Owner.Input.Host, ex.Input.Length);

                //if (Owner.Binarize)
                //{
                //    //Owner.Input.Host.ForEach(v => v >= 127 ? 255 : 0);
                //}

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
