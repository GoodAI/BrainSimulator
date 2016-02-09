using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using System.Drawing;
using GoodAI.Core.Memory;
using YAXLib;

namespace GoodAI.Core.Observers
{
    public class MySpikeRasterObserver : MyAbstractMemoryBlockObserver
    {
        [MyBrowsable, Category("Data")]
        [YAXSerializableField(DefaultValue = -1), YAXElementFor("Structure")]
        public int CustomCount 
        { 
            get { return m_customCount; }
            set 
            {
                if (value < 1)
                    value = -1;  // means disabled
                else if (Target != null)
                {
                    if (value > Target.Count)
                        value = Target.Count; // must not be greater then the observed memory block count

                    // adjust offset if necessary to stay within bounds
                    if (m_offset > Target.Count - value)
                        m_offset = Target.Count - value;
                }

                m_customCount = value;
                TriggerReset(); 
            } 
        }
        private int m_customCount = -1;  // this value is shown in the property list (and is updated on execution) (TODO(Premek))

        [MyBrowsable, Category("Data")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("Structure")]
        public int Offset
        {
            get { return m_offset; }
            set
            {
                if (value < 0)
                    throw new ArgumentException("Offset must not be negative.");
                else if (Target != null)
                {
                    if (value >= Target.Count)
                        value = Target.Count - 1;  // Offset must be within the data

                    if ((m_customCount > 0) && (value > Target.Count - m_customCount))
                    {
                        // adjust CustomCount to stay within the memory block boundaries (lower it)
                        m_customCount = Target.Count - value;
                    }
                }  

                m_offset = value;
                TriggerReset();
            }
        }
        private int m_offset = 0;

        private int Count
        {
            get
            {
                if (m_customCount > 0)
                    return m_customCount;
                else if (Target != null)
                    return Target.Count - m_offset;
                else
                    return 100;  // should not happen
            }
        }

        [YAXSerializableField(DefaultValue = 0.0f)]
        [MyBrowsable, Category("Data"), DisplayName("M\tinValue")]
        public float MinValue { get; set; }

        [YAXSerializableField(DefaultValue = 1.0f)]
        [MyBrowsable, Category("Data")]
        public float MaxValue { get; set; }

        [MyBrowsable, Category("Display")]
        [YAXSerializableField(DefaultValue = 200), YAXElementFor("Structure")]
        [Description("Number of steps in history diplayed (formely X SIZE)")]
        public int NumberOfSteps
        {
            get { return m_numberOfSteps; }
            set 
            {
                m_numberOfSteps = PutIntWithinBounds(value, 2, 10000);  // ought to be enough for anybody ?-)
                TriggerReset(); 
            } 
        }
        private int m_numberOfSteps = 200;

        [MyBrowsable, Category("Display")]
        [YAXSerializableField(DefaultValue = 20), YAXElementFor("Structure")]
        public int GridStep
        {
            get { return m_gridStep; }
            set
            {
                m_gridStep = PutIntWithinBounds(value, 2, 10000);
            }
        }
        private int m_gridStep = 20;

        [YAXSerializableField]
        [MyBrowsable, Category("Texture"), DisplayName("Coloring Method")]
        public RenderingMethod Method { get; set; }

        [YAXSerializableField]
        private uint MARKER_COLOR = 0xFFFFFF80;  // light yellow
        [YAXSerializableField]
        private uint BACKGROUND_COLOR_1 = 0xFF202020;  // very dark gray
        [YAXSerializableField]
        private uint BACKGROUND_COLOR_2 = 0xFF000000;
        [YAXSerializableField]
        private uint GRID_COLOR = 0xFF585858;

        [MyBrowsable, Category("\tVisualization")]
        [Description("Marker")]
        public Color MarkerColor
        {
            get { return Color.FromArgb((int)MARKER_COLOR); }
            set { MARKER_COLOR = (uint)value.ToArgb(); }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Background color one")]
        public Color Background1
        {
            get { return Color.FromArgb((int)BACKGROUND_COLOR_1); }
            set { BACKGROUND_COLOR_1 = (uint)value.ToArgb(); }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Background color two")]
        public Color Background2
        {
            get { return Color.FromArgb((int)BACKGROUND_COLOR_2); }
            set { BACKGROUND_COLOR_2 = (uint)value.ToArgb(); }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Grid color")]
        public Color GridColor
        {
            get { return Color.FromArgb((int)GRID_COLOR); }
            set { GRID_COLOR = (uint)value.ToArgb(); }
        }

        private int m_ringArrayStart;
        private int m_gridStepCounter;

        public MySpikeRasterObserver()
        {
            MinValue = 0.0f;
            MaxValue = 1.0f;
            Method = RenderingMethod.RedGreenScale;

            m_ringArrayStart = 0;
            m_gridStepCounter = 0;

            m_kernel = MyKernelFactory.Instance.Kernel(@"Observers\SpikeRasterObserverKernel");
            TargetChanged += MySpikeRasterObserver_TargetChanged;
        }

        void MySpikeRasterObserver_TargetChanged(object sender, PropertyChangedEventArgs e)
        {
            TriggerReset();
        }        

        protected override void Execute()
        {
            m_kernel.SetConstantVariable("D_COUNT", Count);
            m_kernel.SetConstantVariable("D_X_SIZE", NumberOfSteps);
            m_kernel.SetConstantVariable("D_BACKGROUND_COLOR_1", BACKGROUND_COLOR_1);
            m_kernel.SetConstantVariable("D_BACKGROUND_COLOR_2", BACKGROUND_COLOR_2);
            m_kernel.SetConstantVariable("D_MARKER_COLOR", MARKER_COLOR);
            m_kernel.SetConstantVariable("D_GRID_STEP", GridStep);
            m_kernel.SetConstantVariable("D_GRID_COLOR", GRID_COLOR);

            m_kernel.SetupExecution(Count);

            m_kernel.Run(
                VBODevicePointer,
                Target,
                Offset,
                m_ringArrayStart,
                m_gridStepCounter,
                (int)Method,
                MinValue,
                MaxValue
                );

            m_ringArrayStart = (m_ringArrayStart + 1) % NumberOfSteps;
            m_gridStepCounter = (m_gridStepCounter + 1) % GridStep;
        }
        
        protected override void Reset()
        {
            base.Reset();
            m_ringArrayStart = 0;
            m_gridStepCounter = 0;

            CheckOffsetAndCustomCount();

            TextureHeight = Count;
            TextureWidth = NumberOfSteps;
        }

        private void CheckOffsetAndCustomCount()
        {
            if (Target == null)
                return;

            // check if offset and custom counts are not out of bounds
            if ((m_customCount > 0) && (m_offset + m_customCount > Target.Count))
            {
                m_offset = Math.Max(0, Target.Count - m_customCount);  // first try to lower offset
                m_customCount = Target.Count - m_offset;  // if it is not enough, lower also custom count
            }
            else if (m_offset >= Target.Count)
            {
                m_offset = Target.Count - 1;
            }
        }

        // TODO(premek): move elsewhere
        private static int PutIntWithinBounds(int value, int minimum, int maximum)
        {
            return Math.Min(Math.Max(value, minimum), maximum);
        }
    }
}
