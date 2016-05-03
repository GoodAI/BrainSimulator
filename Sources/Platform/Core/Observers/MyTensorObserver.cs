using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using System.Drawing;
using System.Reflection;
using GoodAI.Core.Memory;
using YAXLib;


namespace GoodAI.Core.Observers
{

    /// <author>GoodAI</author>
    /// <meta>jv</meta>
    /// <status>experimental</status>
    /// <summary>
    /// MemoryBlockObserver which supports better observation of tensors (user can setup tile width, height and no. of tiles in one row).
    /// Could be done better (automatic read from TensorDimensions and manuall override if needed), but better than nothing.
    /// </summary>
    /// <description>
    /// </description>
    [YAXSerializeAs("MemoryBlockObserver")]
    public class MyTensorObserver : MyMemoryBlockObserver
    {
        #region Parameters

        [YAXSerializableField(DefaultValue = false)]
        [MyBrowsable, Category("Tensor Observer"), Description("Enables/disables (computation expensive) tensor observing (if disabled, this is normal MemoryBlockObserver)")]
        public bool ObserveTensors { get; set; }

        private int m_tw, m_th, m_tin;
        [YAXSerializableField(DefaultValue = 20)]
        [MyBrowsable, Category("Tensor Observer")]
        [Description("Width of one tile")]
        public int TileWidth
        {
            get
            {
                return m_tw;
            }
            set
            {
                if (value > 0)
                {
                    m_tw = value;
                }
            }
        }

        [YAXSerializableField(DefaultValue = 20)]
        [MyBrowsable, Category("Tensor Observer")]
        [Description("Height of one tile")]
        public int TileHeight
        {
            get
            {
                return m_th;
            }
            set
            {
                if (value > 0)
                {
                    m_th = value;
                }
            }
        }

        [YAXSerializableField(DefaultValue = 20)]
        [MyBrowsable, Category("Tensor Observer")]
        [Description("Number of tiles displayed in one row")]
        public int TilesInRow
        {
            get
            {
                return m_tin;
            }
            set
            {
                if (value > 0)
                {
                    m_tin = value;
                }
            }
        }

        #endregion

        protected MyCudaKernel m_tiledKernel;

        protected virtual void MyMemoryBlockObserver_TargetChanged(object sender, PropertyChangedEventArgs e)
        {
            base.MyMemoryBlockObserver_TargetChanged(sender, e);

            Type type = Target.GetType().GenericTypeArguments[0];
            m_tiledKernel = MyKernelFactory.Instance.Kernel(@"Observers\ColorScaleObserverTiled" + type.Name);
        }

        protected override void Execute()
        {
            if (ObserveTensors)
            {
                m_tiledKernel.SetupExecution(TextureSize);
                m_tiledKernel.Run(Target.GetDevicePtr(ObserverGPU, 0, TimeStep), Elements, MaxValue, VBODevicePointer, TextureSize, TileWidth, TileHeight, TilesInRow);
                return;
            }
            else
            {
                base.Execute();
            }
        }

        protected override void SetTextureDimensions()
        {
            if (ObserveTensors)
            {
                if (Elements / (TileWidth * TileHeight) != 0)
                {
                    MyLog.WARNING.WriteLine("Memory block '{0}: {1}' observer: {2}", Target.Owner.Name, Target.Name, "Dims.Count not divisible by TileWidth and TileHeight, ignoring");
                    base.SetTextureDimensions();
                    return;
                }
                if (TileWidth * TileHeight * TilesInRow > Elements)
                {
                    TilesInRow = Elements / (TileWidth * TileHeight);
                }
                TextureWidth = TileWidth * TilesInRow;
                TextureHeight = Elements / (TileWidth * TilesInRow);
            }
            else
            {
                base.SetTextureDimensions();
                return;
            }
        }
    }
}
