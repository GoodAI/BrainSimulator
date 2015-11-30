using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using YAXLib;
using OpenTK.Graphics.OpenGL;
using PixelFormat = System.Drawing.Imaging.PixelFormat;

namespace GoodAI.Core.Observers
{
    public class TimePlotObserver : MyAbstractMemoryBlockObserver
    {
        private const int MaxCurveCount = 50;

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

        // TODO(HonzaS): Can this be done in a better way? Watch out for serialization.
        [YAXSerializableField]
        private Color m_colorBackground = Color.White;
        [YAXSerializableField]
        private Color m_colorFont = Color.Black;
        [YAXSerializableField]
        private Color m_colorCurve1 = Color.Red;
        [YAXSerializableField]
        private Color m_colorCurve2 = Color.Blue;
        [YAXSerializableField]
        private Color m_colorCurve3 = Color.Green;
        [YAXSerializableField]
        private Color m_colorCurve4 = Color.Yellow;
        [YAXSerializableField]
        private Color m_colorCurve5 = Color.Purple;
        [YAXSerializableField]
        private Color m_colorCurve6 = Color.Cyan;
        [YAXSerializableField]
        private Color m_colorCurveExtra = Color.Black;
        [YAXSerializableField]

        [MyBrowsable, Category("\tVisualization")]
        [Description("Background color")]
        public Color ColorBackground
        {
            get { return m_colorBackground; }
            set
            {
                m_colorBackground = value;
                m_isDirty = true;
            }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Font color")]
        public Color ColorFont
        {
            get { return m_colorFont; }
            set
            {
                m_colorFont = value;
                m_isDirty = true;
            }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Background color")]
        public Color ColorCurve1
        {
            get { return m_colorCurve1; }
            set
            {
                m_colorCurve1 = value;
                m_isDirty = true;
            }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Curve 2 color")]
        public Color ColorCurve2
        {
            get { return m_colorCurve2; }
            set
            {
                m_colorCurve2 = value;
                m_isDirty = true;
            }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Curve 3 color")]
        public Color ColorCurve3
        {
            get { return m_colorCurve3; }
            set
            {
                m_colorCurve3 = value;
                m_isDirty = true;
            }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Curve 4 color")]
        public Color ColorCurve4
        {
            get { return m_colorCurve4; }
            set
            {
                m_colorCurve4 = value;
                m_isDirty = true;
            }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Curve 5 color")]
        public Color ColorCurve5
        {
            get { return m_colorCurve5; }
            set
            {
                m_colorCurve5 = value;
                m_isDirty = true;
            }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Curve 6 color")]
        public Color ColorCurve6
        {
            get { return m_colorCurve6; }
            set
            {
                m_colorCurve6 = value;
                m_isDirty = true;
            }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Curve other")]
        public Color ColorCurveExtra
        {
            get { return m_colorCurveExtra; }
            set
            {
                m_colorCurveExtra = value;
                m_isDirty = true;
            }
        }

        #endregion // Colors

        private double[] m_valuesHistory;

        private bool m_isDirty;
        private int m_currentRealTimeStep;
        private int m_currentSamplingTimeStep;
        private uint m_lastSimulationStep;

        private int m_currentCursorPosition;

        private int m_plotAreaWidth;
        private int m_plotAreaHeight;
        private int m_plotAreaOffsetX;
        private int m_plotAreaOffsetY;


        private double m_plotCurrentValueMin = double.NaN;
        private double m_plotCurrentValueMax = double.NaN;

        // Used by SCALE method
        private int m_scaleFactor;
        private Bitmap m_bitmap;
        private SolidBrush m_backgroundBrush;
        private int m_samplingCounter;
        private SolidBrush m_textBrush;
        private SolidBrush m_cursorBrush;

        public TimePlotObserver() //constructor with node parameter
        {
            TextureWidth = 800;
            TextureHeight = 400;
            m_plotAreaOffsetX = 100;
            m_plotAreaOffsetY = 0;

            Period = 1;
            Count = 1;

            TargetChanged += MyTimePlotObserver_TargetChanged;
            m_textBrush = new SolidBrush(m_colorFont);
            m_textBrush = new SolidBrush(m_colorFont);
            m_cursorBrush = new SolidBrush(Color.Gray);
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

                m_plotCurrentValueMin = m_boundMin;
                m_plotCurrentValueMax = m_boundMax;
            }

            if (!m_manualBoundHaveBeenSet)
            {
                m_manualBoundMin = m_boundMin;
                m_manualBoundMax = m_boundMax;
            }
        }

        private void UpdateHistoryBuffer()
        {
            if (Count == 0)
                return;

            if (Count > MaxCurveCount)
            {
                MyLog.ERROR.WriteLine("Number of displayed curved is too high (" + Count + ", max " + MaxCurveCount + ")");
                return;
            }

            // Allocate the history
            m_valuesHistory = new double[Count];
        }


        protected override void Reset()
        {
            base.Reset();

            m_plotAreaWidth = TextureWidth - m_plotAreaOffsetX;
            m_plotAreaHeight = TextureHeight - m_plotAreaOffsetY;
            m_isDirty = true;
            m_currentRealTimeStep = 0;
            m_currentSamplingTimeStep = 0;

            m_currentCursorPosition = 0;

            m_backgroundBrush = new SolidBrush(m_colorBackground);

            m_samplingCounter = 0;
            m_scaleFactor = 1;

            UpdateHistoryBuffer();

            // TODO(HonzaS): Investigate why the vertical size must be like this. Probably a <= vs < somewhere.
            m_bitmap = new Bitmap(TextureWidth, TextureHeight - 1, PixelFormat.Format32bppRgb);
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

            Target.SafeCopyToHost();

            bool mustBeUpdated = false;

            var oldValueMin = m_plotCurrentValueMin;
            var oldValueMax = m_plotCurrentValueMax;

            bool newBounds = false;

            if (BoundPolicy == MyBoundPolicy.AUTO)
            {
                // Update the new min / max
                if (Target.Count == 0)
                    return;

                for (int c = 0; c < Count; c++)
                {
                    double value = 0.0;
                    Target.GetValueAt(ref value, c * Stride + Offset);
                    if (m_isDirty)
                    {
                        if (double.IsNaN(value))
                        {
                            // Cant decide bounds
                            return;
                        }

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
            }
            else if (BoundPolicy == MyBoundPolicy.MANUAL)
            {
                m_plotCurrentValueMin = BoundMin;
                m_plotCurrentValueMax = BoundMax;
            }

            using (Graphics graphics = Graphics.FromImage(m_bitmap))
            {
                if (m_isDirty)
                {
                    DrawBackground(graphics);
                    DrawCoordinates(graphics);
                }
                else if (newBounds)
                {
                    DrawScaledHistoryY(graphics, oldValueMin, oldValueMax);
                    DrawCoordinates(graphics);
                }

                PlotValues(graphics, DisplayMethod);
            }

            DisplayBitmap();

            m_isDirty = false;
            if (newValueIsAvailable)
                m_currentRealTimeStep++;
            m_lastSimulationStep = SimulationStep;
        }

        private void PlotValues(Graphics graphics, MyDisplayMethod displayMethod)
        {
            switch (displayMethod)
            {
                case MyDisplayMethod.SCROLL:
                    DrawScroll(graphics);
                    break;
                case MyDisplayMethod.SCALE:
                    DrawScale(graphics);
                    break;
                case MyDisplayMethod.CYCLE:
                default:
                    DrawCycle(graphics);
                    break;
            }
        }

        private double GetCurrentValue(int i)
        {
            //double currentValue = Target.Host[i*Stride + Offset];
            double currentValue = 0.0;
            Target.GetValueAt(ref currentValue, i * Stride + Offset);
            return currentValue;
        }

        private Color GetCurveColor(int i)
        {
            // TODO(HonzaS): Refactoring, see the TODO near the definition of m_colorCurve1.
            Color color;
            switch (i)
            {
                case 0:
                    color = m_colorCurve1;
                    break;
                case 1:
                    color = m_colorCurve2;
                    break;
                case 2:
                    color = m_colorCurve3;
                    break;
                case 3:
                    color = m_colorCurve4;
                    break;
                case 4:
                    color = m_colorCurve5;
                    break;
                case 5:
                    color = m_colorCurve6;
                    break;
                default:
                    color = m_colorCurveExtra;
                    break;
            }
            return color;
        }

        private int ValueToScale(double value)
        {
            double range = m_plotCurrentValueMax - m_plotCurrentValueMin;

            return (int)(m_plotAreaHeight - m_plotAreaHeight * ((value - m_plotCurrentValueMin) / range)) + m_plotAreaOffsetY;
        }

        private void DrawScroll(Graphics graphics)
        {
            throw new NotImplementedException("Scroll is not implemented yet");
        }

        private void DrawCycle(Graphics graphics)
        {
            var backgroundBrush = new SolidBrush(m_colorBackground);

            m_currentCursorPosition++;
            if (m_currentCursorPosition >= m_plotAreaWidth)
                m_currentCursorPosition -= m_plotAreaWidth;

            int x = m_currentCursorPosition + m_plotAreaOffsetX;

            graphics.FillRectangle(backgroundBrush, x, m_plotAreaOffsetY, 1, m_plotAreaHeight);

            for (int curveIndex = 0; curveIndex < Count; curveIndex++)
            {
                var brush = new SolidBrush(GetCurveColor(curveIndex));

                double currentValue = GetCurrentValue(curveIndex);

                double lastValue = m_valuesHistory[curveIndex];

                int y = ValueToScale(currentValue);
                int lastY = ValueToScale(lastValue);

                graphics.FillRectangle(brush, x, Math.Min(y, lastY), 1, Math.Max(1, Math.Abs(y - lastY)));
                graphics.FillRectangle(brush, 0, 400, m_bitmap.Width, 1);

                // Cursor
                graphics.FillRectangle(m_cursorBrush, x+1, m_plotAreaOffsetY, 1, m_plotAreaHeight);

                m_valuesHistory[curveIndex] = currentValue;
            }
        }

        private void DrawScale(Graphics graphics)
        {
            m_samplingCounter++;
            if (m_samplingCounter >= m_scaleFactor)
            {
                m_samplingCounter = 0;
                m_currentCursorPosition++;
            }

            if (m_currentCursorPosition >= m_plotAreaWidth)
            {
                m_currentCursorPosition = m_plotAreaWidth/2;
                m_scaleFactor *= 2;
                DrawScaledHistoryX(graphics, m_plotAreaWidth/2);
            }

            if (m_samplingCounter == 0)
            {
                int x = m_currentCursorPosition + m_plotAreaOffsetX;

                var cursorBrush = new SolidBrush(Color.Gray);

                graphics.FillRectangle(m_backgroundBrush, x, m_plotAreaOffsetY, 1, m_plotAreaHeight);

                for (int curveIndex = 0; curveIndex < Count; curveIndex++)
                {
                    var brush = new SolidBrush(GetCurveColor(curveIndex));

                    double currentValue = GetCurrentValue(curveIndex);
                    double lastValue = m_valuesHistory[curveIndex];

                    int y = ValueToScale(currentValue);
                    int lastY = ValueToScale(lastValue);

                    graphics.FillRectangle(brush, x, Math.Min(y, lastY), 1, Math.Max(1, Math.Abs(y - lastY)));

                    // Cursor
                    graphics.FillRectangle(cursorBrush, x+1, m_plotAreaOffsetY, 1, m_plotAreaHeight);

                    m_valuesHistory[curveIndex] = currentValue;
                }
            }
        }

        private void DrawScaledHistoryX(Graphics graphics, int width)
        {
            DrawScaledHistory(graphics, width, m_plotCurrentValueMin, m_plotCurrentValueMax);
        }

        private void DrawScaledHistoryY(Graphics graphics, double oldValueMin, double oldValueMax)
        {
            DrawScaledHistory(graphics, m_plotAreaWidth, oldValueMin, oldValueMax);
        }

        private void DrawScaledHistory(Graphics graphics, int newWidth, double oldValueMin, double oldValueMax)
        {
            int newHeight = (int) ((oldValueMax - oldValueMin) / (m_plotCurrentValueMax - m_plotCurrentValueMin) * m_plotAreaHeight);
            if (newHeight <= 0)
                return;

            Bitmap targetBitmap = new Bitmap(newWidth, newHeight);

            Rectangle sourceArea = new Rectangle(m_plotAreaOffsetX, m_plotAreaOffsetY, m_plotAreaWidth, m_plotAreaHeight);
            Rectangle targetArea = new Rectangle(0, 0, newWidth, newHeight);

            ScaleBitmap(ref targetBitmap, targetArea, sourceArea);

            graphics.FillRectangle(m_backgroundBrush, sourceArea);

            int heightOffset = (int) (m_plotAreaOffsetY + (m_plotCurrentValueMax - oldValueMax) / (m_plotCurrentValueMax - m_plotCurrentValueMin) * m_plotAreaHeight);

            graphics.DrawImage(targetBitmap, new Rectangle(m_plotAreaOffsetX, heightOffset, newWidth, newHeight));
        }

        private void ScaleBitmap(ref Bitmap targetBitmap, Rectangle targetArea, Rectangle sourceArea)
        {
            using (var scalingGraphics = Graphics.FromImage(targetBitmap))
            {
                scalingGraphics.CompositingMode = CompositingMode.SourceCopy;
                scalingGraphics.CompositingQuality = CompositingQuality.HighQuality;
                scalingGraphics.InterpolationMode = InterpolationMode.NearestNeighbor;
                scalingGraphics.SmoothingMode = SmoothingMode.HighQuality;
                scalingGraphics.PixelOffsetMode = PixelOffsetMode.HighQuality;
                scalingGraphics.DrawImage(m_bitmap, targetArea, sourceArea, GraphicsUnit.Pixel);
            }
        }

        private void DisplayBitmap()
        {
            BitmapData bitmapData = m_bitmap.LockBits(new Rectangle(0, 0, m_bitmap.Width, m_bitmap.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppRgb);

            GL.BindTexture(TextureTarget.Texture2D, TextureId);
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba8, m_bitmap.Width, m_bitmap.Height, 0, OpenTK.Graphics.OpenGL.PixelFormat.Bgra, PixelType.UnsignedByte, bitmapData.Scan0);

            m_bitmap.UnlockBits(bitmapData);
        }

        private void DrawCoordinates(Graphics graphics)
        {
            // Redraw the background
            graphics.FillRectangle(m_backgroundBrush, 0, 0, m_plotAreaOffsetX, m_plotAreaHeight);
            graphics.FillRectangle(m_cursorBrush, m_plotAreaOffsetX-2, 0, 1, m_plotAreaHeight);

            var axisFont = new Font(FontFamily.GenericMonospace, 14);

            // Ordinates
            double range = m_plotCurrentValueMax - m_plotCurrentValueMin;
            double scale = Math.Floor(Math.Log10(range));
            double unit = Math.Pow(10, scale) / 2;
            int displayPrecision = (scale >= 1) ? 0 : (1 - (int)scale);
            double firstOrdinate = Math.Ceiling(m_plotCurrentValueMin / unit) * unit;
            for (int n = 0; firstOrdinate + n * unit < m_plotCurrentValueMax; n++)
            {
                double value = firstOrdinate + n * unit;
                int y = ValueToScale(value);

                string template = Math.Abs(value) >= 100000
                    ? "{0:E" + displayPrecision + "}"
                    : "{0:F" + displayPrecision + "}";

                graphics.DrawString(string.Format(template, value), axisFont, m_textBrush, 0, y - axisFont.Height/2);
                graphics.FillRectangle(m_cursorBrush, m_plotAreaOffsetX-4, y, 4, 1);
            }
        }

        private void DrawBackground(Graphics graphics)
        {
            graphics.FillRectangle(new SolidBrush(m_colorBackground), 0, 0, m_bitmap.Width, m_bitmap.Height);
        }

        public override string GetTargetName(MyNode declaredOwner)
        {
            if (declaredOwner == Target.Owner)
                return Target.Owner.Name + ": " + Target.Name;

            return declaredOwner.Name + " (" + Target.Owner.Name + "): " + Target.Name;
        }

        protected override string CreateTargetIdentifier()
        {
            if (Target != null)
                return Target.Owner.Id + "#" + Target.Name;
            else
                return String.Empty;
        }

        public override void RestoreTargetFromIdentifier(MyProject project)
        {
            if (TargetIdentifier == null)
                return;

            string[] split = TargetIdentifier.Split('#');
            if (split.Length != 2)
                return;

            MyWorkingNode node = (MyWorkingNode)project.GetNodeById(int.Parse(split[0]));

            if (node != null)
                Target = (MyMemoryBlock<float>)MyMemoryManager.Instance.GetMemoryBlockByName(node, split[1]);
        }

        protected override void OnCountChanged()
        {
        }
    }
}
