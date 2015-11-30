using GoodAI.Core.Memory;
using GoodAI.Core.Observers.Helper;
using GoodAI.Core.Utils;
using ManagedCuda;
using System;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Imaging;
using GoodAI.Core.Nodes;
using OpenTK.Graphics.OpenGL;
using YAXLib;
using PixelFormat = System.Drawing.Imaging.PixelFormat;

namespace GoodAI.Core.Observers
{
    public class MatrixObserver : MyAbstractMemoryBlockObserver
    {
        [YAXSerializableField] private int m_decimalCount;

        [MyBrowsable, Category("Display"), Description("Number of displayed decimals")]
        public int DecimalCount
        {
            get { return m_decimalCount; }
            set
            {
                if (value < 0)
                    return;

                m_decimalCount = value;
                TriggerReset();
            }
        }

        [MyBrowsable, Category("Display"), Description("Number of columns in the memory block")]
        public int ColumnCount
        {
            get { return m_columnCount; }
            set
            {
                if (value <= 0)
                    return;

                m_columnCount = value;
                TriggerReset();
            }
        }

        [YAXSerializableField(DefaultValue = false)] protected bool m_crop = false;

        [MyBrowsable, Category("Crop"), Description("Enable cropping"), DisplayName("\tCropping")]
        public bool Crop
        {
            get { return m_crop; }
            set
            {
                m_crop = value;
                TriggerReset();
            }
        }

        [YAXSerializableField(DefaultValue = 0)] protected int m_xStart = 0;

        [MyBrowsable, Category("Crop"), Description("Starting column"), DisplayName("\tXStart")]
        public int XStart
        {
            get { return m_xStart; }
            set
            {
                if (value < 0)
                    return;
                if (value >= m_columnCount)
                    value = m_columnCount - 1;

                if (value + m_xLength > m_columnCount)
                    m_xLength = m_columnCount - value;

                m_xStart = value;
                TriggerReset();
            }
        }

        [YAXSerializableField(DefaultValue = 0)] protected int m_xLength = 0;

        [MyBrowsable, Category("Crop"), Description("Number of columns to display")]
        public int XLength
        {
            get { return m_xLength; }
            set
            {
                if (value <= 0)
                    return;
                if (value > m_columnCount)
                    value = m_columnCount;

                if (m_xStart + value > m_columnCount)
                    m_xStart = m_columnCount - value;

                m_xLength = value;
                TriggerReset();
            }
        }

        [YAXSerializableField(DefaultValue = 0)] protected int m_yStart = 0;

        [MyBrowsable, Category("Crop"), Description("Starting column"), DisplayName("\tYStart")]
        public int YStart
        {
            get { return m_yStart; }
            set
            {
                if (value < 0)
                    return;
                if (value >= m_rowCount)
                    value = m_rowCount - 1;

                if (value + m_yLength > m_rowCount)
                    m_yLength = m_rowCount - value;

                m_yStart = value;
                TriggerReset();
            }
        }

        [YAXSerializableField(DefaultValue = 0)] protected int m_yLength = 0;

        [MyBrowsable, Category("Crop"), Description("Number of rows to display")]
        public int YLength
        {
            get { return m_yLength; }
            set
            {
                if (value <= 0)
                    return;
                if (value > m_rowCount)
                    value = m_rowCount;

                if (m_yStart + value > m_rowCount)
                    m_yStart = m_rowCount - value;

                m_yLength = value;
                TriggerReset();
            }
        }

        private int m_columnCount;
        private int m_rowCount;
        private int m_nbValues;

        private Font m_font;
        private int m_characterMargin;
        private Bitmap m_bitmap;
        private int m_frameMarginPx;
        private int m_characterMarginRightPx;
        private int m_characterMarginTopPx;
        private Brush m_backgroundBrush;
        private Brush m_textBrush;
        private Brush m_negativeTextBrush;
        private int m_cellWidth;
        private int m_cellHeight;
        private Type m_valueType;

        public MatrixObserver() //constructor with node parameter
        {
            DecimalCount = 2;
            TargetChanged += MyMatrixObserver_TargetChanged;
            m_characterMargin = 1; // In characters.
            m_frameMarginPx = 10;
            m_characterMarginRightPx = 2;
            m_characterMarginTopPx = 2;
        }

        void MyMatrixObserver_TargetChanged(object sender, PropertyChangedEventArgs e)
        {
            if (Target == null)
                return;

            m_valueType = Target.GetType().GenericTypeArguments[0];
            if (m_valueType != typeof (float)
                && m_valueType != typeof (double)
                && m_valueType != typeof (decimal))
            {
                DecimalCount = 0;
            }
        }

        protected override void Execute()
        {
            if (m_columnCount*m_rowCount == 0)
                return;

            using (Graphics graphics = Graphics.FromImage(m_bitmap))
            {
                DrawBackground(graphics);

                Target.SafeCopyToHost();

                DrawMatrix(graphics);
            }

            DisplayBitmap();
        }

        private void DrawBackground(Graphics graphics)
        {
            graphics.FillRectangle(m_backgroundBrush, 0, 0, m_bitmap.Width, m_bitmap.Height);
        }

        private void DrawMatrix(Graphics graphics)
        {
            for (int y = 0; y < YLength; y++)
                for (int x = 0; x < XLength; x++) {
                {
                    int index = (YStart + y)*m_columnCount + (XStart + x);
                    if (index >= Target.Count)
                        break;

                    var drawX = x*m_cellWidth + m_frameMarginPx/2;
                    var drawY = y*m_cellHeight + m_frameMarginPx/2;

                    double doubleValue = 0d;
                    if (m_valueType == typeof (int))
                    {
                        int value = 0;
                        Target.GetValueAt(ref value, index);
                        doubleValue = Convert.ToDouble(value);
                    }
                    else if (m_valueType == typeof (float))
                    {
                        float value = 0f;
                        Target.GetValueAt(ref value, index);
                        doubleValue = Convert.ToDouble(value);
                    }
                    else if (m_valueType == typeof (double))
                    {
                        Target.GetValueAt(ref doubleValue, index);
                    }

                    DrawCellContent(graphics, doubleValue, drawX, drawY);
                }
            }
        }

        private void DrawCellContent(Graphics graphics, double value, int drawX, int drawY)
        {
            string template = Math.Abs(value) >= 100000
                ? "{0:E" + DecimalCount + "}"
                : "{0:F" + DecimalCount + "}";

            Brush brush = value >= 0 ? m_textBrush : m_negativeTextBrush;
            graphics.DrawString(string.Format(template, value), m_font, brush, drawX, drawY);
        }

        private void DisplayBitmap()
        {
            BitmapData bitmapData = m_bitmap.LockBits(new Rectangle(0, 0, m_bitmap.Width, m_bitmap.Height),
                ImageLockMode.ReadOnly, PixelFormat.Format32bppRgb);

            GL.BindTexture(TextureTarget.Texture2D, TextureId);
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba8, m_bitmap.Width, m_bitmap.Height, 0,
                OpenTK.Graphics.OpenGL.PixelFormat.Bgra, PixelType.UnsignedByte, bitmapData.Scan0);

            m_bitmap.UnlockBits(bitmapData);
        }

        protected override void Reset()
        {
            base.Reset();

            m_font = new Font(FontFamily.GenericMonospace, 10, FontStyle.Bold);
            m_backgroundBrush = new SolidBrush(Color.White);
            m_textBrush = new SolidBrush(Color.Black);
            m_negativeTextBrush = new SolidBrush(Color.Red);

            SetupDimensions();
            SetupTextureSize();

            m_bitmap = new Bitmap(TextureWidth, TextureHeight - 1, PixelFormat.Format32bppRgb);
        }

        private void SetupTextureSize()
        {
            // Max digits before the dot + margin
            int charactersPerNumber = 5 + m_characterMargin;

            if (DecimalCount != 0)
                charactersPerNumber += DecimalCount + 1; // The decimals plus the dot

            int charactersPerLine = charactersPerNumber*XLength - m_characterMargin;

            // Only used for font size checking. The correct bitmap is constructed later.
            m_bitmap = new Bitmap(1, 1);

            int fontWidth;
            int fontHeight;

            using (var graphics = Graphics.FromImage(m_bitmap))
            {
                RectangleF fontSize = MeasureDisplayStringWidth(graphics, "0", m_font);
                fontWidth = (int) Math.Ceiling(fontSize.Width);
                fontHeight = (int) Math.Ceiling(fontSize.Height);
            }

            m_cellWidth = fontWidth*charactersPerNumber;
            m_cellHeight = fontHeight;

            TextureWidth = fontWidth*charactersPerLine + m_frameMarginPx;
            TextureHeight = fontHeight*YLength + m_frameMarginPx;
        }

        private static RectangleF MeasureDisplayStringWidth(Graphics graphics, string text, Font font)
        {
            StringFormat format = new StringFormat();
            RectangleF rect = new RectangleF(0, 0, 1000, 1000);
            CharacterRange[] ranges = { new CharacterRange(0, text.Length) };
            Region[] regions = new Region[1];

            format.SetMeasurableCharacterRanges(ranges);

            regions = graphics.MeasureCharacterRanges(text, font, rect, format);
            rect = regions[0].GetBounds(graphics);

            return new RectangleF(rect.X, rect.Y, rect.Width + 1.0f, rect.Height);
        }

        private void SetupDimensions()
        {
            m_nbValues = Target.Count;

            if (m_columnCount <= 0)
            {
                if (Target.ColumnHint > 0)
                {
                    m_columnCount = Math.Min(Target.Count, Target.ColumnHint);
                    m_rowCount = (int) Math.Ceiling((float) (m_nbValues)/m_columnCount);
                }
                else
                {
                    m_columnCount = m_nbValues;
                    m_rowCount = 1;
                }
            }

            m_rowCount = (int) Math.Ceiling((float) (m_nbValues)/m_columnCount);

            // If cropping is turned off, automatically adjust to the whole matrix.
            if (!Crop)
            {
                m_xLength = m_columnCount;
                m_yLength = m_rowCount;
            }
            
            if (m_xLength == 0)
                m_xLength = m_columnCount;
            if (m_yLength == 0)
                m_yLength = m_rowCount;
        }
    }
}
