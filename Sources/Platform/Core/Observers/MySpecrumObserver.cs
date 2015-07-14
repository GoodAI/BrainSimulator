using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
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

namespace GoodAI.Core.Observers
{
    [YAXSerializeAs("SpecrumObserver")]
    public class MySpecrumObserver : MyAbstractMemoryBlockObserver
    {
        public enum MyDisplayMethod
        {
            CYCLE,
            SCALE,
            SCROLL
        }

        public enum RenderingMethod
        {
            GrayScale,
            ColorScale,
            BlackWhite
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

        #region Texture
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
        #endregion

        #region Scale
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
        #endregion

        #region Rendering
        [YAXSerializableField(DefaultValue = MyDisplayMethod.CYCLE)]
        private MyDisplayMethod m_displayMethod = MyDisplayMethod.CYCLE;

        [MyBrowsable, Category("\tRendering"), Description("Display method")]
        public MyDisplayMethod DisplayMethod
        {
            get
            {
                return m_displayMethod;
            }

            set
            {
                m_displayMethod = value;
                TriggerReset();
            }
        }
        #endregion

        private CudaDeviceVariable<float> m_valuesHistory;
        private MyMemoryBlock<float> m_history{get; set;}

        private int m_currentTimeStep;
        private int m_numColumns = 100;
        private int currentColumn = 0;



        public MySpecrumObserver()
        {
            Method = RenderingMethod.ColorScale;
            BoundPolicy = MyBoundPolicy.INHERITED;
            MinValue = 0;
            MaxValue = 1;

            TargetChanged += MyMemoryBlockObserver_TargetChanged;
        }

        void MyMemoryBlockObserver_TargetChanged(object sender, PropertyChangedEventArgs e)
        {
            Type type = Target.GetType().GenericTypeArguments[0];
            m_kernel = MyKernelFactory.Instance.Kernel(@"Observers\ColorScaleObserver" + type.Name);
        }

        protected override void Execute()
        {
            m_currentTimeStep = (int)SimulationStep;
            currentColumn = (int)SimulationStep % m_numColumns;

            if ((Target is MyMemoryBlock<float>) && m_history != null)
            {
                //m_history.CopyFromMemoryBlock((Target as MyMemoryBlock<float>), 0, 0, Target.Count);
            }
            

            m_kernel.SetupExecution(TextureSize);
            m_kernel.Run(Target, 5, (int)Scale, MinValue, MaxValue, VBODevicePointer, TextureSize);
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

            m_valuesHistory = new CudaDeviceVariable<float>(Target.Count * m_numColumns);
            m_valuesHistory.Memset(0);

            //m_history = MyMemoryManager.Instance.CreateMemoryBlock<float>(Target.Owner.Parent);
            //m_history.Count = Target.Count * m_numColumns;

            SetDefaultTextureDimensions(Target.Count);
        }

        protected override void SetDefaultTextureDimensions(int pixelCount)
        {
            TextureWidth = m_numColumns;
            TextureHeight = pixelCount;
        }
    }
}
