using BrainSimulator.Memory;
using BrainSimulator.Utils;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using YAXLib;

namespace BrainSimulator.Observers
{
    public class MyHistogramObserver : MyAbstractMemoryBlockObserver
    {
        public enum Option
        {
            True,
            False
        }

        [MyBrowsable, Category("Parameters")]
        [YAXSerializableField(DefaultValue = Option.False), YAXElementFor("Structure")]
        public Option SET_BOUNDARIES { get; set; }

        [MyBrowsable, Category("Data")]
        [YAXSerializableField(DefaultValue = 1.00f), YAXElementFor("Structure")]
        public float MAX_VALUE
        {
            get
            {
                return maxValue;
            }
            set
            {
                maxValue = value;
                BINS = (int)Math.Ceiling((double)((maxValue - MIN_VALUE) / BIN_VALUE_WIDTH)) + 2;
                TriggerReset();
            }
        }

        private float maxValue = 1.00f;

        [MyBrowsable, Category("Data")]
        [YAXSerializableField(DefaultValue = 0.00f), YAXElementFor("Structure")]
        public float MIN_VALUE
        {
            get
            {
                return minValue;
            }
            set
            {
                minValue = value;
                BINS = (int)Math.Ceiling((double)((MAX_VALUE - minValue) / BIN_VALUE_WIDTH)) + 2;
                TriggerReset();
            }
        }

        private float minValue = 0.00f;

        private int m_bins;
        [MyBrowsable, Category("Data")]
        [YAXSerializableField(DefaultValue = 12), YAXElementFor("Structure")]
        public int BINS
        {
            get { return m_bins; }
            private set
            {
                m_bins = value;
                TriggerReset();
            }
        }

        [MyBrowsable, Category("Data")]
        [YAXSerializableField(DefaultValue = 0.10f), YAXElementFor("Structure")]
        public float BIN_VALUE_WIDTH
        {
            get
            {
                return binValueWidth;
            }

            set
            {
                binValueWidth = value;
                BINS = (int)Math.Ceiling((double)((MAX_VALUE - MIN_VALUE) / binValueWidth)) + 2;
            }
        }

        private float binValueWidth = 0.10f;

        private int m_binPixelWidth;
        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 32), YAXElementFor("Structure")]
        public int BIN_PIXEL_WIDTH
        {
            get { return m_binPixelWidth; }
            set
            {
                m_binPixelWidth = value;
                TriggerReset();
            }
        }

        private int m_binPixelHeight;
        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 250), YAXElementFor("Structure")]
        public int BIN_PIXEL_HEIGHT
        {
            get { return m_binPixelHeight; }
            set
            {
                m_binPixelHeight = value;
                TriggerReset();
            }
        }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 100), YAXElementFor("Structure")]
        public int UPDATE_STEP { get; set; }

        [YAXSerializableField]
        private uint COLOR_ONE = 0xFF00CCFF;

        [YAXSerializableField]
        private uint COLOR_TWO = 0xFF00B8E6;

        [YAXSerializableField]
        private uint BACKGROUND = 0xFFFFFFFF;

        [YAXSerializableField]
        private uint OUT_OF_BOUNDS = 0xFFFF0000;

        [MyBrowsable, Category("\tVisualization")]
        [Description("Color one")]
        public Color ColorOne
        {
            get { return Color.FromArgb((int)COLOR_ONE); }
            set { COLOR_ONE = (uint)value.ToArgb(); }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Color two")]
        public Color ColorTwo
        {
            get { return Color.FromArgb((int)COLOR_TWO); }
            set { COLOR_TWO = (uint)value.ToArgb(); }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Background color")]
        public Color BackgroundColor
        {
            get { return Color.FromArgb((int)BACKGROUND); }
            set { BACKGROUND = (uint)value.ToArgb(); }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Out of bounds color")]
        public Color OutOfBoundsColor
        {
            get { return Color.FromArgb((int)OUT_OF_BOUNDS); }
            set { OUT_OF_BOUNDS = (uint)value.ToArgb(); }
        }
        private MyCudaKernel m_computeHistogram;
        private MyCudaKernel m_visualizeHistogram;

        private CudaDeviceVariable<int> m_d_HistogramData;

        public MyHistogramObserver()
        {
            MAX_VALUE = 1.00f;
            MIN_VALUE = 0.00f;

            BIN_VALUE_WIDTH = 0.10f;

            BINS = (int)Math.Ceiling((double)((MAX_VALUE - MIN_VALUE) / BIN_VALUE_WIDTH)) + 2;

            BIN_PIXEL_HEIGHT = 250;
            BIN_PIXEL_WIDTH = 32;

            UPDATE_STEP = 100;

            COLOR_ONE = 0xFF00CCFF;
            COLOR_TWO = 0xFF00B8E6;
            BACKGROUND = 0xFFFFFFFF;
            OUT_OF_BOUNDS = 0xFFFF0000;

            m_computeHistogram = MyKernelFactory.Instance.Kernel(@"Observers\ComputeHistogramKernel");
            m_visualizeHistogram = MyKernelFactory.Instance.Kernel(@"Observers\VisualizeHistogramKernel");
        }

        protected override void Execute()
        {
            if (SimulationStep % UPDATE_STEP == 0 || SimulationStep == 1)
            {
                m_computeHistogram.DynamicSharedMemory = (uint)((int)sizeof(int) * BINS);

                m_computeHistogram.SetConstantVariable("D_BINS", BINS);
                m_computeHistogram.SetConstantVariable("D_MAX_VALUE", MAX_VALUE);
                m_computeHistogram.SetConstantVariable("D_MIN_VALUE", MIN_VALUE);
                m_computeHistogram.SetConstantVariable("D_BIN_VALUE_WIDTH", BIN_VALUE_WIDTH);
                m_computeHistogram.SetConstantVariable("D_MEMORY_BLOCK_SIZE", Target.Count);

                m_d_HistogramData.Memset(0);
                m_computeHistogram.SetupExecution(
                    Target.Count
                    );

                m_computeHistogram.Run(
                    Target,
                    m_d_HistogramData.DevicePointer
                    );

                m_visualizeHistogram.SetConstantVariable("D_BINS", BINS);
                m_visualizeHistogram.SetConstantVariable("D_BIN_PIXEL_WIDTH", BIN_PIXEL_WIDTH);
                m_visualizeHistogram.SetConstantVariable("D_BIN_PIXEL_HEIGHT", BIN_PIXEL_HEIGHT);
                m_visualizeHistogram.SetConstantVariable("D_COLOR_ONE", COLOR_ONE);
                m_visualizeHistogram.SetConstantVariable("D_COLOR_TWO", COLOR_TWO);
                m_visualizeHistogram.SetConstantVariable("D_COLOR_BACKGROUND", BACKGROUND);
                m_visualizeHistogram.SetConstantVariable("D_OUT_OF_BOUNDS", OUT_OF_BOUNDS);

                m_visualizeHistogram.SetupExecution(
                    new dim3(BIN_PIXEL_WIDTH, 1, 1),
                    new dim3(BINS, 1, 1)
                    );

                m_visualizeHistogram.Run(
                    m_d_HistogramData.DevicePointer,
                    VBODevicePointer
                    );

                if (SET_BOUNDARIES == Option.True)
                {
                    Target.SafeCopyToHost();
                    float min = (Target as MyMemoryBlock<float>).Host.Min();
                    float max = (Target as MyMemoryBlock<float>).Host.Max();
                    if (min == max)
                    {
                        max = min + 1.00f;
                    }
                    MAX_VALUE = max;
                    MIN_VALUE = min;
                    SET_BOUNDARIES = Option.False;
                    TriggerReset();
                }
            }
        }

        protected override void Reset()
        {
            TextureHeight = BIN_PIXEL_HEIGHT;
            TextureWidth = BIN_PIXEL_WIDTH * BINS;

            if (m_d_HistogramData != null)
            {
                m_d_HistogramData.Dispose();
            }
            m_d_HistogramData = new CudaDeviceVariable<int>(BINS);
            m_d_HistogramData.Memset(0);
        }
    }
}