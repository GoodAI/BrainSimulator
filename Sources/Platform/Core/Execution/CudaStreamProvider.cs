using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Utils;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace GoodAI.Core.Execution
{
    public class CudaStreamProvider
    {
        public static CudaStreamProvider Instance { get; } = new CudaStreamProvider();

        public CudaStream CurrentStream => m_streamPool[m_currentIndex];

        private const int StreamsPerThread = 8;

        private readonly CudaStream[] m_streamPool = new CudaStream[StreamsPerThread];

        private int m_currentIndex = 0;

        private CudaStreamProvider()
        {
            for (var i = 0; i < StreamsPerThread; i++)
            {
                m_streamPool[i] = new CudaStream();
            }
        }

        public void SwitchToNextStream()
        {
            m_currentIndex = (m_currentIndex + 1) % StreamsPerThread;

            //MyLog.DEBUG.WriteLine($"Switched to cuda stream #{m_currentIndex}");
        }
    }
}
