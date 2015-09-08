using GoodAI.Core;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Layers;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Observers
{
    class MyRBMFilterObserver : MyNodeObserver<MyAbstractWeightLayer>
    {

        [YAXSerializableField(DefaultValue = 28)]
        private int m_filterWidth;

        [MyBrowsable, Category("Display")]
        public int FilterWidth
        {
            get { return m_filterWidth; }
            set
            {
                m_filterWidth = value;
                TriggerReset();
            }
        }

        [YAXSerializableField(DefaultValue = 28)]
        private int m_filterHeight;

        [MyBrowsable, Category("Display")]
        public int FilterHeight
        {
            get { return m_filterHeight; }
            set
            {
                m_filterHeight = value;
                TriggerReset();
            }
        }


        [YAXSerializableField(DefaultValue = 1)]
        private int m_columns;

        [MyBrowsable, Category("Display")]
        public int Columns
        {
            get { return m_columns; }
            set
            {
                if (value >= 1 && value <= Target.Neurons)
                    m_columns = value;
                TriggerReset();
            }
        }

        [YAXSerializableField(DefaultValue = -5)]
        private float m_minValue;

        [MyBrowsable, Category("Bounds"), DisplayName("\tMinValue")]
        public float MinValue
        {
            get { return m_minValue; }
            set
            {
                m_minValue = value;
            }
        }

        [YAXSerializableField(DefaultValue = 5)]
        private float m_maxValue;

        [MyBrowsable, Category("Bounds")]
        public float MaxValue
        {
            get { return m_maxValue; }
            set
            {
                m_maxValue = value;
            }
        }

        public MyRBMFilterObserver()
        {
            m_maxValue = 5;
            m_minValue = -5;
            m_columns = 1;
            m_filterHeight = 0;
            m_filterWidth = 0;
            m_kernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"RBM\RBMKernels", "RBMFilterObserver");
            if (Target != null && Target.Output != null)
                m_columns = (int)Math.Sqrt(Target.Output.Count);

            TriggerReset();
        }

        protected override void Execute()
        {
            m_kernel.SetupExecution(Target.Weights.Count);
            m_kernel.Run(Target.Weights, Target.Output, Target.Neurons, FilterWidth, FilterHeight,
                 MinValue, MaxValue, TextureWidth, TextureHeight, VBODevicePointer);
        }        

        protected override void Reset()
        {
            if (Columns < 1)
                Columns = 1;

            if (FilterHeight == 0 || FilterWidth == 0)
            {
                FilterHeight = (int)Math.Sqrt(Target.Weights.Count / Target.Neurons);
                FilterWidth = (int)Math.Sqrt(Target.Weights.Count / Target.Neurons);

            }

            //Set texture size, it will trigger texture buffer reallocation
            TextureWidth = Columns * FilterWidth;
            if (Target.Neurons % Columns == 0)
                TextureHeight = Target.Neurons / Columns * FilterHeight;
            else
                TextureHeight = (1 + (Target.Neurons / Columns)) * FilterHeight;

        }
    }
}
