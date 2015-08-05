using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers.Helper;
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
    [YAXSerializeAs("SpectrumObserver")]
    public class MySpectrumObserver : MyAbstractMemoryBlockObserver
    {

        [YAXSerializableField]
        private uint COLOR_FONT = 0xFFFF0000; // White
        [YAXSerializableField]
        private uint COLOR_BACKGROUND = 0xFF000000; // Black

        private CudaDeviceVariable<uint> m_canvas;
        private CudaDeviceVariable<float> m_valuesHistory;
        private CudaDeviceVariable<float> m_HistoryDeviceBuffer;

        private int m_numRows;
        private int m_numColumns;
        private int m_currentTimeStep;
        private int currentColumn = 0;

        public MySpectrumObserver()
        {
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

            int maxValue = 0;
            for (int i = 0; i < Target.Count / 2; i++)
            {
                float val = 0;
                Target.GetValueAt<float>(ref val, i);
                m_valuesHistory[((m_numRows - i) * m_numColumns) + currentColumn] = val;

                if (val > maxValue)
                    maxValue = (int)val;
            }

            m_kernel.SetupExecution(TextureSize);
            m_kernel.Run(m_valuesHistory.DevicePointer, 5, 0, 0, maxValue, VBODevicePointer, TextureSize);

            /*
            String text = "Spectrogram";
            int colIdx = 0;
            foreach (char c in text)
            {
                m_HistoryDeviceBuffer[10 + colIdx] = (float)(c - ' ');
                colIdx += 1;
            }
            MyDrawStringHelper.DrawStringFromGPUMem(m_HistoryDeviceBuffer, 0, (MyDrawStringHelper.CharacterHeight + 1), 0, 0x999999, VBODevicePointer, TextureWidth, TextureHeight, 0, 0);*/
            //MyDrawStringHelper.DrawString(text, 0, 0, COLOR_BACKGROUND, COLOR_FONT, m_valuesHistory.DevicePointer, TextureWidth, TextureHeight);
            
        }

        private void drawCoordinates()
        {
            // Set a blank canvas
            m_canvas = new CudaDeviceVariable<uint>(VBODevicePointer);

            //m_canvas.Memset(COLOR_BACKGROUND);
            

            /*
            double m_plotCurrentValueMax = 256;
            double m_plotCurrentValueMin = 0;

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
                double y = TextureHeight - 256 * (value - m_plotCurrentValueMin) / range - MyDrawStringHelper.CharacterHeight / 2;
                MyDrawStringHelper.DrawString(valueStr, 0, (int)y, COLOR_BACKGROUND, COLOR_FONT, VBODevicePointer, TextureWidth, TextureHeight);
            }*/

        }

        protected override void Reset()
        {
            base.Reset();

            m_numRows = (Target.Count / 2) - 1;
            m_numColumns = Target.Count * 2;

            // Allocate the history
            m_valuesHistory = new CudaDeviceVariable<float>((Target.Count / 2) * m_numColumns);
            m_valuesHistory.Memset(0);

            // Allocate the history
            m_HistoryDeviceBuffer = new CudaDeviceVariable<float>(m_numRows * m_numColumns);
            m_HistoryDeviceBuffer.Memset(0);

            SetDefaultTextureDimensions(Target.Count);
        }

        protected override void SetDefaultTextureDimensions(int pixelCount)
        {
            TextureWidth = m_numColumns;
            TextureHeight = pixelCount / 2;
        }
    }
}
