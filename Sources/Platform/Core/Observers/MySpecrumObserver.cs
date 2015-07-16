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

        private CudaDeviceVariable<float> m_valuesHistory;

        private int m_numRows;
        private int m_numColumns;
        private int m_currentTimeStep;
        private int currentColumn = 0;

        public MySpecrumObserver()
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

            //m_valuesHistory.Memset(0);
            int maxValue = 0;
            for (int i = 0; i < Target.Count; i++)
            {
                float val = 0;
                Target.GetValueAt<float>(ref val, i);
                m_valuesHistory[((m_numRows - i) * m_numRows) + currentColumn] = val;

                if (val > maxValue)
                    maxValue = (int)val;
            }

            m_kernel.SetupExecution(TextureSize);
            m_kernel.Run(m_valuesHistory.DevicePointer, 5, 0, 0, maxValue, VBODevicePointer, TextureSize);
        }

        protected override void Reset()
        {
            base.Reset();

            m_numRows = Target.Count - 1;
            m_numColumns = Target.Count - 1;

            // Allocate the history
            m_valuesHistory = new CudaDeviceVariable<float>(Target.Count * m_numColumns);
            m_valuesHistory.Memset(0);

            SetDefaultTextureDimensions(Target.Count);
        }

        protected override void SetDefaultTextureDimensions(int pixelCount)
        {
            TextureWidth = m_numColumns;
            TextureHeight = pixelCount;
        }
    }
}
