using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers.Helper;
using GoodAI.Core.Utils;
using ManagedCuda;
using System;
using System.ComponentModel;
using System.Drawing;
using YAXLib;

namespace GoodAI.Core.Observers
{
    public class MyTimePlotObserver : MyObserver<MyMemoryBlock<float>>
    {
        public const int nbCurvesMax = 50;

        public enum MyDisplayMethod
        {
            CYCLE,
            SCALE,
            SCROLL
        }

        public enum MyBoundPolicy
        {
            INHERITED,
            AUTO,
            MANUAL
        }

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

        [YAXSerializableField(DefaultValue = MyBoundPolicy.INHERITED)]
        private MyBoundPolicy m_boundPolicy = MyBoundPolicy.INHERITED;

        [MyBrowsable, Category("\tRendering"), Description("The way the Min / Max of Y axis are chosen")]
        public MyBoundPolicy BoundPolicy
        {
            get
            {
                return m_boundPolicy;
            }

            set
            {
                // Before set
                if (m_boundPolicy == MyBoundPolicy.MANUAL)
                {
                    m_manualBoundMin = m_boundMin;
                    m_manualBoundMax = m_boundMax;
                }

                // Set
                m_boundPolicy = value;

                // After Set
                if (m_boundPolicy == MyBoundPolicy.INHERITED)
                {
                    if (float.IsInfinity(Target.MinValueHint) || float.IsInfinity(Target.MaxValueHint))
                    {
                        m_boundMin = -1;
                        m_boundMax = 1;
                        m_boundPolicy = MyBoundPolicy.AUTO;
                        MyLog.WARNING.WriteLine("At least one of the inherited bounds is infinite. Switch to AUTO mode.");
                    }
                    else
                    {
                        m_boundMin = Target.MinValueHint;
                        m_boundMax = Target.MaxValueHint;
                    }
                }
                else if (m_boundPolicy == MyBoundPolicy.MANUAL)
                {
                    m_boundMin = m_manualBoundMin;
                    m_boundMax = m_manualBoundMax;
                }
                m_isDirty = true;
            }
        }

        [YAXSerializableField(DefaultValue = -1)]
        private float m_boundMin;


        [MyBrowsable, Category("\tRendering"), Description("Manually set the lowest value.\n/!\\ != min value")]
        public float BoundMin
        {
            get
            {
                return m_boundMin;
            }

            set
            {
                if (value >= BoundMax || float.IsInfinity(value))
                    return;
                m_boundPolicy = MyBoundPolicy.MANUAL;
                m_manualBoundHaveBeenSet = true;
                m_boundMin = value;
                m_isDirty = true;
            }
        }

        [YAXSerializableField(DefaultValue = 1)]
        private float m_boundMax;

        [MyBrowsable, Category("\tRendering"), Description("Manually set the highest value\n/!\\ != max value")]
        public float BoundMax
        {
            get
            {
                return m_boundMax;
            }

            set
            {
                if (value <= BoundMin || float.IsInfinity(value))
                    return;
                m_boundPolicy = MyBoundPolicy.MANUAL;
                m_manualBoundHaveBeenSet = true;
                m_boundMax = value;
                m_isDirty = true;
            }
        }

        [YAXSerializableField(DefaultValue = -1)]
        private float m_manualBoundMin;
        [YAXSerializableField(DefaultValue = 1)]
        private float m_manualBoundMax;
        [YAXSerializableField(DefaultValue = false)]
        private bool m_manualBoundHaveBeenSet;


        private int m_observerWidth = 500;
        [YAXSerializableField(DefaultValue = 500)]
        [MyBrowsable, Category("\tRendering"), Description("Texture width")]
        public int ObserverWidth
        {
            get
            {
                return m_observerWidth;
            }

            set
            {
                if (value <= 0)
                    return;
                m_observerWidth = value;
                TextureWidth = value + m_plotAreaOffsetX;
                TriggerReset();
            }
        }

        [YAXSerializableField(DefaultValue = 200)]
        [MyBrowsable, Category("\tRendering"), Description("Texture height")]
        public int ObserverHeight
        {
            get
            {
                return TextureHeight;
            }

            set
            {
                if (value <= 10)
                    return;
                TextureHeight = value;
                TriggerReset();
            }
        }


        [YAXSerializableField(DefaultValue = 1)]
        private int m_period = 1;

        [MyBrowsable, Category("\tSampling"), Description("Sampling Rate")]
        public int Period
        {
            get
            {
                return m_period;
            }

            set
            {
                if (value <= 0)
                    return;
                m_period = value;
                TriggerReset();
            }
        }


        [YAXSerializableField(DefaultValue = 0)]
        private int m_delay = 0;

        [MyBrowsable, Category("\tSampling"), Description("Delayed measure (number of skipped time Period)")]
        public int Delay
        {
            get
            {
                return m_delay;
            }

            set
            {
                if (value < 0)
                    return;
                m_delay = value;
                TriggerReset();
            }
        }

        [YAXSerializableField(DefaultValue = 0)]
        private int m_offset = 0;
        [MyBrowsable, Category("\tSampling"), Description("Offset of the value to plot")]
        public int Offset
        {
            get { return m_offset; }
            set
            {
                if (value < 0 || (Target != null && (Stride * (Count - 1) + value) > Target.Count))
                    return;
                m_offset = value;
                TriggerReset();
            }
        }

        [YAXSerializableField(DefaultValue = 1)]
        private int m_stride = 1;
        [MyBrowsable, Category("\tSampling"), Description("Offset of the value to plot")]
        public int Stride
        {
            get { return m_stride; }
            set
            {
                if (value < 1 || (Target != null && (value * (Count - 1) + Offset) > Target.Count))
                    return;
                m_stride = value;
                TriggerReset();
            }
        }

        [YAXSerializableField(DefaultValue = 1)]   
        private int m_count = 1;
        [MyBrowsable, Category("\tSampling"), Description("Number of values to plot")]
        public int Count 
        {
            get { return m_count; } 
            set
            {
                if (value <= 0 || (Target != null && (Stride * (value - 1) + Offset) > Target.Count))
                    return;
                m_count = value;
                TriggerReset();
            }
        }


        #region CurveColors

        private bool m_colorsAreDirty = true;

        private uint[] COLOR_CURVE_TO_CUDA_ARRAY = new uint[6];
        [YAXSerializableField]
        private uint COLOR_BACKGROUND = 0xFFFFFFFF; // White
        [YAXSerializableField]
        private uint COLOR_FONT = 0xFF000000; // Black
        [YAXSerializableField]
        private uint COLOR_CURVE1 = 0xFFFF0000; // Red
        [YAXSerializableField]
        private uint COLOR_CURVE2 = 0xFF0000FF; // Blue
        [YAXSerializableField]
        private uint COLOR_CURVE3 = 0xFF00FF00; // Green
        [YAXSerializableField]
        private uint COLOR_CURVE4 = 0xFFFFFF00; // Yellow
        [YAXSerializableField]
        private uint COLOR_CURVE5 = 0xFFFF00FF; // Purple
        [YAXSerializableField]
        private uint COLOR_CURVE6 = 0xFF00FFFF; // Cyan
        [YAXSerializableField]
        private uint COLOR_CURVE_EXTRA = 0xFF000000; // Black
        [MyBrowsable, Category("\tVisualization")]
        [Description("Background color")]
        public Color ColorBackground
        {
            get { return Color.FromArgb((int)COLOR_BACKGROUND); }
            set { COLOR_BACKGROUND = (uint)value.ToArgb(); m_colorsAreDirty = true; }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Font color")]
        public Color ColorFont
        {
            get { return Color.FromArgb((int)COLOR_FONT); }
            set { COLOR_FONT = (uint)value.ToArgb(); m_colorsAreDirty = true; }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Background color")]
        public Color ColorCurve1
        {
            get { return Color.FromArgb((int)COLOR_CURVE1); }
            set { COLOR_CURVE1 = (uint)value.ToArgb(); m_colorsAreDirty = true; }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Curve 2 color")]
        public Color ColorCurve2
        {
            get { return Color.FromArgb((int)COLOR_CURVE2); }
            set { COLOR_CURVE2 = (uint)value.ToArgb(); m_colorsAreDirty = true; }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Curve 3 color")]
        public Color ColorCurve3
        {
            get { return Color.FromArgb((int)COLOR_CURVE3); }
            set { COLOR_CURVE3 = (uint)value.ToArgb(); m_colorsAreDirty = true; }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Curve 4 color")]
        public Color ColorCurve4
        {
            get { return Color.FromArgb((int)COLOR_CURVE4); }
            set { COLOR_CURVE4 = (uint)value.ToArgb(); m_colorsAreDirty = true; }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Curve 5 color")]
        public Color ColorCurve5
        {
            get { return Color.FromArgb((int)COLOR_CURVE5); }
            set { COLOR_CURVE5 = (uint)value.ToArgb(); m_colorsAreDirty = true; }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Curve 6 color")]
        public Color ColorCurve6
        {
            get { return Color.FromArgb((int)COLOR_CURVE6); }
            set { COLOR_CURVE6 = (uint)value.ToArgb(); m_colorsAreDirty = true; }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Curve other")]
        public Color ColorCurveExtra
        {
            get { return Color.FromArgb((int)COLOR_CURVE_EXTRA);}
            set { COLOR_CURVE_EXTRA = (uint)value.ToArgb(); m_colorsAreDirty = true; }
        }
        private void UpdateColorsToGpu()
        {
            COLOR_CURVE_TO_CUDA_ARRAY[0] = COLOR_CURVE1;
            COLOR_CURVE_TO_CUDA_ARRAY[1] = COLOR_CURVE2;
            COLOR_CURVE_TO_CUDA_ARRAY[2] = COLOR_CURVE3;
            COLOR_CURVE_TO_CUDA_ARRAY[3] = COLOR_CURVE4;
            COLOR_CURVE_TO_CUDA_ARRAY[4] = COLOR_CURVE5;
            COLOR_CURVE_TO_CUDA_ARRAY[5] = COLOR_CURVE6;
            switch (DisplayMethod)
            {
                case MyDisplayMethod.CYCLE:
                    m_cycleKernel.SetConstantVariable("D_COLOR_BACKGROUND", COLOR_BACKGROUND);
                    m_cycleKernel.SetConstantVariable("D_COLOR_CURVES", COLOR_CURVE_TO_CUDA_ARRAY);
                    m_cycleKernel.SetConstantVariable("D_COLOR_CURVE_EXTRA", COLOR_CURVE_EXTRA);
                    break;

                case MyDisplayMethod.SCALE:
                    m_scaleKernel.SetConstantVariable("D_COLOR_BACKGROUND", COLOR_BACKGROUND);
                    m_scaleKernel.SetConstantVariable("D_COLOR_CURVES", COLOR_CURVE_TO_CUDA_ARRAY);
                    m_scaleKernel.SetConstantVariable("D_COLOR_CURVE_EXTRA", COLOR_CURVE_EXTRA);
                    break;

                case MyDisplayMethod.SCROLL:
                    m_scrollKernel.SetConstantVariable("D_COLOR_BACKGROUND", COLOR_BACKGROUND);
                    m_scrollKernel.SetConstantVariable("D_COLOR_CURVES", COLOR_CURVE_TO_CUDA_ARRAY);
                    m_scrollKernel.SetConstantVariable("D_COLOR_CURVE_EXTRA", COLOR_CURVE_EXTRA);
                    break;

                default:
                    break;
            }
        }
        #endregion // Colors
        
        private CudaDeviceVariable<float> m_valuesHistory;

        private bool m_isDirty;
        private int m_currentRealTimeStep;
        private int m_currentSamplingTimeStep;
        private uint m_lastSimulationStep;
        private CudaDeviceVariable<uint> m_canvas;

        private int m_plotAreaWidth;
        private int m_plotAreaHeight;
        private int m_plotAreaOffsetX;
        private int m_plotAreaOffsetY;


        private double m_plotCurrentValueMin = double.NaN;
        private double m_plotCurrentValueMax = double.NaN;

        // Used by SCROLL method
        MyCudaKernel m_scrollKernel;
        MyCudaKernel m_scrollShiftKernel;

        // Used by SCALE method
        private int m_scaleFactor;
        private float[] m_scaleAverage;
        private int m_scaleAverageWeight;
        private int m_nbValuesSaved;
        private MyCudaKernel m_scaleKernel;
        private MyCudaKernel m_scaleDownScaleKernel;

        // Used by CYCLE method
        private MyCudaKernel m_cycleKernel;
        private MyCudaKernel m_verticalLineKernel;


        public MyTimePlotObserver() //constructor with node parameter
        {
            TextureWidth = 800;
            TextureHeight = 400;
            m_plotAreaOffsetX = 150;
            m_plotAreaOffsetY = 0;

            Period = 1;
            Count = 1;

            TargetChanged += MyTimePlotObserver_TargetChanged;
        }

        void MyTimePlotObserver_TargetChanged(object sender, PropertyChangedEventArgs e)
        {
            //Set appropriate bounds

            if (m_boundPolicy == MyBoundPolicy.INHERITED)
            {
                if (float.IsInfinity(Target.MinValueHint))
                {
                    m_boundMin = -1;
                    m_boundPolicy = MyBoundPolicy.AUTO;
                }
                else
                    m_boundMin = Target.MinValueHint;

                if (float.IsInfinity(Target.MaxValueHint))
                {
                    m_boundMax = 1;
                    m_boundPolicy = MyBoundPolicy.AUTO;
                }
                else
                    m_boundMax = Target.MaxValueHint;
            } 

            if (!m_manualBoundHaveBeenSet)
            {
                m_manualBoundMin = m_boundMin;
                m_manualBoundMax = m_boundMax;
            }
        }


        
        private void updateHistoryBuffer()
        {
            if (Count == 0)
                return;

            if (Count > nbCurvesMax)
            {
                MyLog.ERROR.WriteLine("Number of displayed curved is too high (" + Count + ", max " + nbCurvesMax + ")");
                return;
            }

            if (m_valuesHistory != null)
            {
                m_valuesHistory.Dispose();
            }

            // Allocate the history
            int historySize = m_plotAreaWidth * Count;
            m_valuesHistory = new CudaDeviceVariable<float>(historySize);
            m_valuesHistory.Memset(0);
        }

        public override void Dispose()
        {
            if (m_valuesHistory != null)
                m_valuesHistory.Dispose();
            if (m_canvas != null)
                m_canvas.Dispose();
            base.Dispose();
        }


        protected override void Reset()
        {
            base.Reset();

            m_StringDeviceBuffer = new CudaDeviceVariable<float>(1000);
            m_StringDeviceBuffer.Memset(0);

            switch (DisplayMethod)
            {
                case MyDisplayMethod.CYCLE:
                    m_cycleKernel = MyKernelFactory.Instance.Kernel(@"Observers\PlotObserverCycleKernel", true);
                    m_verticalLineKernel = MyKernelFactory.Instance.Kernel(@"Observers\VerticalLineKernel", true);
                    break;
                
                case MyDisplayMethod.SCALE:
                    m_scaleFactor = 1;
                    m_scaleAverage = new float[Count];
                    for (int i = 0; i < Count; i++)
                        m_scaleAverage[i] = 0;
                    m_scaleAverageWeight = 0;
                    m_nbValuesSaved = 0;
                    m_scaleKernel = MyKernelFactory.Instance.Kernel(@"Observers\PlotObserverScaleKernel", true);
                    m_scaleDownScaleKernel = MyKernelFactory.Instance.Kernel(@"Observers\PlotObserverScaleDownScaleKernel", true);
                    break;
                
                case MyDisplayMethod.SCROLL:
                    m_scrollKernel = MyKernelFactory.Instance.Kernel(@"Observers\PlotObserverScrollKernel", true);
                    m_scrollShiftKernel = MyKernelFactory.Instance.Kernel(@"Observers\PlotObserverScrollShiftKernel", true);
                    break;
                
                default:
                break;
            }

            m_plotAreaWidth = TextureWidth - m_plotAreaOffsetX;
            m_plotAreaHeight = TextureHeight - m_plotAreaOffsetY;
            m_isDirty = true;
            m_currentRealTimeStep = 0;
            m_currentSamplingTimeStep = 0;
            UpdateColorsToGpu();
            updateHistoryBuffer();
        }
        

        protected override void Execute()
        {
            bool newValueIsAvailable = m_lastSimulationStep != SimulationStep;

            if (m_currentRealTimeStep % Period != 0 && newValueIsAvailable)
            {
                m_currentRealTimeStep++;
                return;
            }

            m_currentSamplingTimeStep = (m_currentRealTimeStep / Period);

            if (m_currentSamplingTimeStep < m_delay && newValueIsAvailable)
            {
                m_currentRealTimeStep++;
                return;
            }
            bool mustBeUpdated = false;

            if (BoundPolicy == MyBoundPolicy.AUTO)
            {
                // Update the new min / max
                bool newBounds = false;
                Target.SafeCopyToHost();
                if (Target.Count == 0)
                    return;
                for (int c = 0; c < Count; c++)
                {
                    double value = Target.Host[c * Stride + Offset];
                    if (m_isDirty)
                    {
                        if (double.IsNaN(value))
                        {
                            // Cant decide bounds
                            return;
                        }
                        // First value
                        m_plotCurrentValueMax = value + 0.01f;
                        m_plotCurrentValueMin = value - 0.01f;
                    }
                    else
                    {
                        if (!double.IsNaN(value)) // Change bounds only if values are real
                        {
                            // Next value
                            if (value > m_plotCurrentValueMax)
                            {
                                m_plotCurrentValueMax = value;
                                newBounds = true;
                            }
                            else if (value < m_plotCurrentValueMin)
                            {
                                m_plotCurrentValueMin = value;
                                newBounds = true;
                            }

                            if (newBounds)
                            {
                                double range = m_plotCurrentValueMax - m_plotCurrentValueMin;
                                m_plotCurrentValueMax += range * 0.1;
                                m_plotCurrentValueMin -= range * 0.1;

                                m_boundMin = (float)m_plotCurrentValueMin;
                                m_boundMax = (float)m_plotCurrentValueMax;
                                OnRuntimePropertyChanged();
                            }
                        }
                    }
                }
                mustBeUpdated = newBounds;
            }
            else if (BoundPolicy == MyBoundPolicy.MANUAL)
            {
                m_plotCurrentValueMin = BoundMin;
                m_plotCurrentValueMax = BoundMax;
            }

            //MyLog.DEBUG.WriteLine("min " + m_plotCurrentValueMin + " max " + m_plotCurrentValueMax);


            if (m_isDirty)
            {
                // Set a blank canvas
                m_canvas = new CudaDeviceVariable<uint>(VBODevicePointer);
                mustBeUpdated = true;
            }



            if (m_colorsAreDirty)
                UpdateColorsToGpu();

            if (mustBeUpdated || m_colorsAreDirty)
            {
                drawCoordinates();
                m_colorsAreDirty = false;
                mustBeUpdated = true;
            }

            switch (DisplayMethod)
            {
                case MyDisplayMethod.CYCLE:
                    runMethodCycle(mustBeUpdated);
                    break;
                case MyDisplayMethod.SCALE:
                    runMethodScale(mustBeUpdated);
                    break;
                case MyDisplayMethod.SCROLL:
                    runMethodScroll(mustBeUpdated);
                    break;

                // Add new methods here

                default:
                    throw new NotImplementedException();
            }

            m_isDirty = false;
            if (newValueIsAvailable)
                m_currentRealTimeStep++;
            m_lastSimulationStep = SimulationStep;
        }

        private CudaDeviceVariable<float> m_StringDeviceBuffer;

        private void drawCoordinates()
        {
            m_canvas.Memset(COLOR_BACKGROUND);

            // Ordinates
            double range = m_plotCurrentValueMax - m_plotCurrentValueMin;
            double scale = Math.Floor(Math.Log10(range));
            double unit = Math.Pow(10, scale) / 2;
            int displayPrecision = (scale >= 1) ? 0 : (1 - (int)scale);
            double firstOrdinate = Math.Ceiling(m_plotCurrentValueMin / unit) * unit;
            for (int n = 0; firstOrdinate + n * unit < m_plotCurrentValueMax; n++)
            {
                double value = firstOrdinate + n * unit;
                string valueStr = string.Format("{0,8:N" + displayPrecision + "}", value);
                double y = TextureHeight - m_plotAreaOffsetY - m_plotAreaHeight * (value - m_plotCurrentValueMin) / range - MyDrawStringHelper.CharacterHeight / 2;
                MyDrawStringHelper.String2Index(valueStr, m_StringDeviceBuffer);
                MyDrawStringHelper.DrawStringFromGPUMem(m_StringDeviceBuffer, 0, (int)y, COLOR_BACKGROUND, COLOR_FONT, VBODevicePointer, TextureWidth, TextureHeight, 0, valueStr.Length);
            }

        }


        /*
         * -----------------------
         * METHOD CYCLE
         * -----------------------
         */
        private void runMethodCycle(bool mustBeUpdated)
        {
            m_cycleKernel.SetConstantVariable("D_NB_CURVES",            Count);
            m_cycleKernel.SetConstantVariable("D_TEXTURE_WIDTH",        TextureWidth);
            m_cycleKernel.SetConstantVariable("D_PLOTAREA_WIDTH",       m_plotAreaWidth);
            m_cycleKernel.SetConstantVariable("D_PLOTAREA_HEIGHT",      m_plotAreaHeight);
            m_cycleKernel.SetConstantVariable("D_PLOTAREA_OFFSET_X",    m_plotAreaOffsetX);
            m_cycleKernel.SetConstantVariable("D_MIN_VALUE",            m_plotCurrentValueMin);
            m_cycleKernel.SetConstantVariable("D_MAX_VALUE",            m_plotCurrentValueMax);

            int currentColumn = m_currentSamplingTimeStep % m_plotAreaWidth;

            if (Stride == 1)
            {
                m_valuesHistory.CopyToDevice(Target.GetDevicePtr(this), Offset * sizeof(float), currentColumn * Count * sizeof(float), Count * sizeof(float));
            }
            else
            {
                //not really happy with this one
                for (int i = 0; i < Count; i++)
                {
                    m_valuesHistory.CopyToDevice(Target.GetDevicePtr(this), (i * Stride + Offset) * sizeof(float), (currentColumn * Count + i) * sizeof(float), sizeof(float));
                }
            }

            if (mustBeUpdated)
            {
                // Draw curves
                int nbColumnsToDraw = Math.Min(m_currentSamplingTimeStep + 1, m_plotAreaWidth);
                m_cycleKernel.SetupExecution(nbColumnsToDraw * m_plotAreaHeight);
                m_cycleKernel.Run(
                    VBODevicePointer,
                    5,
                    nbColumnsToDraw,
                    m_valuesHistory.DevicePointer
                    );
            }
            else
            {
                // Draw only the needed columns
                m_cycleKernel.SetupExecution(m_plotAreaHeight);
                m_cycleKernel.Run(
                    VBODevicePointer,
                    currentColumn,
                    1,
                    m_valuesHistory.DevicePointer
                    );
            }

            // Draw a vertical grey line
            m_verticalLineKernel.SetupExecution(m_plotAreaHeight);
            m_verticalLineKernel.Run(
                VBODevicePointer,
                m_plotAreaOffsetX + (currentColumn + 1) % m_plotAreaWidth,
                TextureWidth,
                m_plotAreaHeight
                );
        }

        /*
         * -----------------------
         * METHOD SCALE
         * -----------------------
         */
        private void runMethodScaleAddValue(float[] newValue, ref bool newDataToDraw, ref bool mustBeUpdated)
        {
            int newAverageWeight = m_scaleAverageWeight + 1;
            for (int c = 0; c < Count; c++)
                m_scaleAverage[c] = m_scaleAverage[c] * m_scaleAverageWeight / newAverageWeight + newValue[Offset + c] / newAverageWeight;
            m_scaleAverageWeight = newAverageWeight;
            if (m_scaleAverageWeight == m_scaleFactor)
            {
                // Write the average to the history, and reset the accumulator
                m_scaleAverageWeight = 0;
                int historyIndex = m_nbValuesSaved / m_scaleFactor;
                for (int c = 0; c < Count; c++)
                {
                    m_valuesHistory[historyIndex * Count + c] = m_scaleAverage[c];
                    m_scaleAverage[c] = 0;
                    newDataToDraw = true;
                }

                // If the history is full, we have to downscale
                if (historyIndex == m_valuesHistory.Size / Count - 1)
                {
                    m_scaleFactor *= 2;
                    int size = Count * m_valuesHistory.Size / 2;
                    m_scaleDownScaleKernel.SetupExecution(size);
                    m_scaleDownScaleKernel.Run(
                        m_valuesHistory.DevicePointer,
                        Count,
                        size
                        );
                    drawCoordinates();
                    mustBeUpdated = true;
                }
            }

            m_nbValuesSaved++;
        }


        private void runMethodScale(bool mustBeUpdated)
        {
            m_scaleKernel.SetConstantVariable("D_NB_CURVES", Count);
            m_scaleKernel.SetConstantVariable("D_TEXTURE_WIDTH", TextureWidth);
            m_scaleKernel.SetConstantVariable("D_PLOTAREA_WIDTH", m_plotAreaWidth);
            m_scaleKernel.SetConstantVariable("D_PLOTAREA_HEIGHT", m_plotAreaHeight);
            m_scaleKernel.SetConstantVariable("D_PLOTAREA_OFFSET_X", m_plotAreaOffsetX);
            m_scaleKernel.SetConstantVariable("D_MIN_VALUE", m_plotCurrentValueMin);
            m_scaleKernel.SetConstantVariable("D_MAX_VALUE", m_plotCurrentValueMax);

            bool newDataToDraw = false;

            Target.SafeCopyToHost();

            runMethodScaleAddValue(Target.Host, ref newDataToDraw, ref mustBeUpdated);
           

            if (mustBeUpdated)
            {
                // Draw curves
                int nbColumnsToDraw = m_nbValuesSaved / m_scaleFactor;
                m_scaleKernel.SetupExecution(nbColumnsToDraw * m_plotAreaHeight);
                m_scaleKernel.Run(
                    VBODevicePointer,
                    0,
                    nbColumnsToDraw,
                    m_valuesHistory.DevicePointer
                    );
            }
            else if (newDataToDraw)
            {
                // Draw the last columns
                m_scaleKernel.SetupExecution(m_plotAreaHeight);
                int columnStart = (m_currentSamplingTimeStep / m_scaleFactor);

                m_scaleKernel.Run(
                    VBODevicePointer,
                    columnStart,
                    1,
                    m_valuesHistory.DevicePointer
                    );
            }
            else
            {
                // Nothing to output here
            }
        }


        /*
         * -----------------------
         * METHOD SCROLL
         * -----------------------
         */
        private void runMethodScroll(bool mustBeUpdated)
        {
            m_scrollKernel.SetConstantVariable("D_NB_CURVES", Count);
            m_scrollKernel.SetConstantVariable("D_TEXTURE_WIDTH", TextureWidth);
            m_scrollKernel.SetConstantVariable("D_PLOTAREA_WIDTH", m_plotAreaWidth);
            m_scrollKernel.SetConstantVariable("D_PLOTAREA_HEIGHT", m_plotAreaHeight);
            m_scrollKernel.SetConstantVariable("D_PLOTAREA_OFFSET_X", m_plotAreaOffsetX);
            m_scrollKernel.SetConstantVariable("D_MIN_VALUE", m_plotCurrentValueMin);
            m_scrollKernel.SetConstantVariable("D_MAX_VALUE", m_plotCurrentValueMax);

            int currentColumn = m_currentSamplingTimeStep % m_plotAreaWidth;

            // No timestep was skipped, no need to interpolate
            m_valuesHistory.CopyToDevice(Target.GetDevicePtr(this), Offset * sizeof(float), currentColumn * Count * sizeof(float), Count * sizeof(float));

            if (mustBeUpdated)
            {
                // Draw curves
                int nbColumsToDraw;
                if (m_currentSamplingTimeStep >= m_plotAreaWidth)
                    nbColumsToDraw = m_plotAreaWidth;
                else
                    nbColumsToDraw = m_currentSamplingTimeStep + 1;
                m_scrollKernel.SetupExecution(nbColumsToDraw * m_plotAreaHeight);

                m_scrollKernel.Run(
                    VBODevicePointer,
                    (true ? 1 : 0), // Render everything
                    m_currentSamplingTimeStep,
                    m_valuesHistory.DevicePointer
                    );
            }
            else
            {
                if (m_currentSamplingTimeStep >= m_plotAreaWidth)
                {
                    // Shift all the pixels one pixel to the left
                    m_scrollShiftKernel.SetConstantVariable("D_TEXTURE_WIDTH", TextureWidth);
                    m_scrollShiftKernel.SetConstantVariable("D_PLOTAREA_WIDTH", m_plotAreaWidth);
                    m_scrollShiftKernel.SetConstantVariable("D_PLOTAREA_HEIGHT", m_plotAreaHeight);
                    m_scrollShiftKernel.SetConstantVariable("D_PLOTAREA_OFFSET_X", m_plotAreaOffsetX);
                    m_scrollShiftKernel.SetupExecution(m_plotAreaHeight);
                    m_scrollShiftKernel.Run(
                        VBODevicePointer
                        );
                }

                // Draw only the needed columns
                m_scrollKernel.SetupExecution(m_plotAreaHeight);

                m_scrollKernel.Run(
                    VBODevicePointer,
                    (false ? 1 : 0), // Render only 1 column
                    m_currentSamplingTimeStep,
                    m_valuesHistory.DevicePointer
                    );
            }
        }

        public override string GetTargetName(MyNode declaredOwner)
        {
            if (declaredOwner == Target.Owner)
            {
                return Target.Owner.Name + ": " + Target.Name;
            }
            else
            {
                return declaredOwner.Name + " (" + Target.Owner.Name + "): " + Target.Name;
            }
        }

        protected override string CreateTargetIdentifier()
        {
            if (Target != null)
            {
                return Target.Owner.Id + "#" + Target.Name;
            }
            else return String.Empty;
        }

        public override void RestoreTargetFromIdentifier(MyProject project)
        {
            if (TargetIdentifier != null)
            {
                string[] split = TargetIdentifier.Split('#');
                if (split.Length == 2)
                {
                    MyWorkingNode node = (MyWorkingNode)project.GetNodeById(int.Parse(split[0]));

                    if (node != null)
                    {
                        Target = (MyMemoryBlock<float>)MyMemoryManager.Instance.GetMemoryBlockByName(node, split[1]);
                    }
                }
            }
        }
    }
}
