using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using GoodAI.Core.Memory;
using YAXLib;

namespace GoodAI.Core.Observers
{
    [YAXSerializeAs("MemoryBlockObserver")]
    public class MyMemoryBlockObserver : MyAbstractMemoryBlockObserver
    {        
        public enum MyBoundPolicy
        {
            Inherited,
            Manual
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
                m_methodSelected = true;
                TriggerReset(); 
            }
        }
        private RenderingMethod m_method;

        private bool m_methodSelected;

        [YAXSerializableField]
        [MyBrowsable, Category("Scale + Bounds")]
        public RenderingScale Scale { get; set; }

        [YAXSerializableField(DefaultValue = MyBoundPolicy.Manual)]
        private MyBoundPolicy m_boundPolicy = MyBoundPolicy.Inherited;

        [MyBrowsable, Category("Scale + Bounds"), Description("The way the Min / Max of Y axis are chosen")]
        public MyBoundPolicy BoundPolicy
        {
            get { return m_boundPolicy; }

            set
            {                
                m_boundPolicy = value;
                if (m_boundPolicy == MyBoundPolicy.Inherited)
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
                BoundPolicy = MyBoundPolicy.Manual; 
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
                BoundPolicy = MyBoundPolicy.Manual;
            }
        }

        [YAXSerializableField(DefaultValue = 2)]
        [MyBrowsable, Category("Texture"), DisplayName("Vector Elements")]
        public int Elements { get; set; }

        [YAXSerializableField(DefaultValue = 0)]
        [MyBrowsable, Category("Temporal")]
        public int TimeStep { get; set; }

        protected MyCudaKernel m_vectorKernel;
        protected MyCudaKernel m_rgbKernel;

        public MyMemoryBlockObserver()
        {
            BoundPolicy = MyBoundPolicy.Inherited;
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

            if (m_methodSelected)
                return;

            m_methodSelected = true;

            string colorScheme;
            RenderingMethod renderingMethod;
            if (Target.Metadata.TryGetValue(MemoryBlockMetadataKeys.RenderingMethod, out colorScheme) &&
                Enum.TryParse(colorScheme, out renderingMethod))
            {
                Method = renderingMethod;
                return;
            }

            // The default when no hint is provided.
            Method = RenderingMethod.RedGreenScale;
        }

        protected override void Execute()
        {
            if (Method == RenderingMethod.Vector)
            {
                m_vectorKernel.SetupExecution(TextureSize);
                m_vectorKernel.Run(Target.GetDevicePtr(ObserverGPU, 0, TimeStep), Elements, MaxValue, VBODevicePointer, TextureSize);
            }
            else if (Method == RenderingMethod.RGB)
            {
                m_rgbKernel.SetupExecution(TextureSize);
                m_rgbKernel.Run(Target.GetDevicePtr(ObserverGPU, 0, TimeStep), VBODevicePointer, TextureSize);
            }
            else
            {
                m_kernel.SetupExecution(TextureSize);                
                m_kernel.Run(Target.GetDevicePtr(ObserverGPU, 0, TimeStep), (int)Method, (int)Scale, MinValue, MaxValue, VBODevicePointer, TextureSize);
            }
        }

        private void ResetBounds()
        {
            if (BoundPolicy == MyBoundPolicy.Inherited)
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
