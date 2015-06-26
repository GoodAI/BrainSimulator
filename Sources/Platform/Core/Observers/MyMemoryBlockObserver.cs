using BrainSimulator.Memory;
using BrainSimulator.Nodes;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace BrainSimulator.Observers
{
    [YAXSerializeAs("MemoryBlockObserver")]
    public class MyMemoryBlockObserver : MyAbstractMemoryBlockObserver
    {        
        public enum RenderingMethod
        {           
            RedGreenScale,
            GrayScale,
            ColorScale,
            BlackWhite,
            Vector,
            Raw,
            RGB
        }

        public enum MyBoundPolicy
        {
            INHERITED,
            MANUAL
        }

        public enum RenderingScale
        {
            Linear,
            InvTan
        }

        [YAXSerializableField]
        [MyBrowsable, Category("Texture"), DisplayName("Coloring Method")]
        public RenderingMethod Method 
        { 
            get { return m_method; }
            set 
            { 
                m_method = value; 
                TriggerReset(); 
            }
        }

        private RenderingMethod m_method;

        [YAXSerializableField]
        [MyBrowsable, Category("Scale + Bounds")]
        public RenderingScale Scale { get; set; }

        [YAXSerializableField(DefaultValue = MyBoundPolicy.MANUAL)]
        private MyBoundPolicy m_boundPolicy = MyBoundPolicy.INHERITED;

        [MyBrowsable, Category("Scale + Bounds"), Description("The way the Min / Max of Y axis are chosen")]
        public MyBoundPolicy BoundPolicy
        {
            get { return m_boundPolicy; }

            set
            {                
                m_boundPolicy = value;
                if (m_boundPolicy == MyBoundPolicy.INHERITED)
                {
                    TriggerReset();
                }
            }
        }

        [YAXSerializableField(DefaultValue = 0)]
        private float m_minValue;

        [MyBrowsable, Category("Scale + Bounds"), DisplayName("M\tinValue")]
        public float MinValue 
        {
            get { return m_minValue; }
            set 
            { 
                m_minValue = value;
                BoundPolicy = MyBoundPolicy.MANUAL; 
            }
        }

        [YAXSerializableField(DefaultValue = 1)]
        private float m_maxValue;

        [MyBrowsable, Category("Scale + Bounds")]
        public float MaxValue
        {
            get { return m_maxValue; }
            set
            {
                m_maxValue = value;
                BoundPolicy = MyBoundPolicy.MANUAL;
            }
        }

        [YAXSerializableField(DefaultValue = 2)]
        [MyBrowsable, Category("Texture"), DisplayName("Vector Elements")]
        public int Elements { get; set; }

        private MyCudaKernel m_vectorKernel;
        private MyCudaKernel m_rgbKernel;

        public MyMemoryBlockObserver()
        {
            Method = RenderingMethod.RedGreenScale;
            BoundPolicy = MyBoundPolicy.INHERITED;
            Elements = 2;
            MinValue = 0;
            MaxValue = 1;

            TargetChanged += MyMemoryBlockObserver_TargetChanged;
        }

        void MyMemoryBlockObserver_TargetChanged(object sender, PropertyChangedEventArgs e)
        {
            Type type = Target.GetType().GenericTypeArguments[0];
            m_kernel = MyKernelFactory.Instance.Kernel(@"Observers\ColorScaleObserver" + type.Name);
            m_vectorKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Observers\ColorScaleObserverSingle", "DrawVectorsKernel");
            m_rgbKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Observers\ColorScaleObserverSingle", "DrawRGBKernel");              
        }

        protected override void Execute()
        {                   
            if (Method == RenderingMethod.Vector)
            {
                m_vectorKernel.SetupExecution(TextureSize);
                m_vectorKernel.Run(Target, Elements, MaxValue, VBODevicePointer, TextureSize);
            }
            else if (Method == RenderingMethod.RGB)
            {
                m_rgbKernel.SetupExecution(TextureSize);
                m_rgbKernel.Run(Target, VBODevicePointer, TextureSize);
            }
            else
            {
                m_kernel.SetupExecution(TextureSize);                
                m_kernel.Run(Target, (int)Method, (int)Scale, MinValue, MaxValue, VBODevicePointer, TextureSize);
            }
        }

        private void ResetBounds()
        {
            if (BoundPolicy == MyBoundPolicy.INHERITED)
            {
                if (!float.IsNegativeInfinity(Target.MinValueHint))
                    m_minValue = Target.MinValueHint;
                else
                    m_minValue = 0;
                if (!float.IsPositiveInfinity(Target.MaxValueHint))
                    m_maxValue = Target.MaxValueHint;
                else
                    m_maxValue = 1;
            }
        }

        protected override void Reset()
        {
            base.Reset();
            ResetBounds();

            if (Method == RenderingMethod.Vector)
            {
                SetDefaultTextureDimensions(Target.Count / Elements);
            }
            else if (Method == RenderingMethod.RGB)
            {
                SetDefaultTextureDimensions(Target.Count / 3);
            }
            else
            {
                SetDefaultTextureDimensions(Target.Count);
            }
        }        

        protected override void SetDefaultTextureDimensions(int pixelCount)  
        {
            if (Target.ColumnHint > 1)
            {
                TextureWidth = Target.ColumnHint;
                TextureHeight =
                    pixelCount % Target.ColumnHint == 0 ?
                    pixelCount / Target.ColumnHint :
                    pixelCount / Target.ColumnHint + 1;
            }
            else
            {
                base.SetDefaultTextureDimensions(pixelCount);
            }
        }        
    }
}
