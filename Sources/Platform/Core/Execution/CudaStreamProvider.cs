using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Utils;
using GoodAI.Platform.Core.Profiling;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace GoodAI.Core.Execution
{
    internal sealed class CudaStreamProvider : IDisposable
    {
        public static CudaStreamProvider Instance { get; } = new CudaStreamProvider();

        public CudaStream CurrentStream => m_streamPool[m_currentIndex];

        private const int StreamsPerThread = 16;

        private readonly CudaStream[] m_streamPool = new CudaStream[StreamsPerThread];

        private int m_currentIndex = 0;

        private readonly LoggingStopwatch m_stopwatch = new LoggingStopwatch("SyncAllStreams", 100) { Enabled = false };

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

            MyLog.DEBUG.WriteLine($"Switched to cuda stream #{m_currentIndex}");
        }

        public void SynchronizeAllStreams()
        {
            m_stopwatch.Start();

            // (Synchronizing the whole context would take roughtly the same time.)
            foreach (var cudaStream in m_streamPool)
            {
                cudaStream.Synchronize();
            }

            m_stopwatch.StopAndSometimesPrintStats();
        }

        private void ReleaseUnmanagedResources()
        {
            foreach (var cudaStream in m_streamPool)
                cudaStream.Dispose();
        }

        public void Dispose()
        {
            ReleaseUnmanagedResources();
            GC.SuppressFinalize(this);
        }

        ~CudaStreamProvider()
        {
            ReleaseUnmanagedResources();
        }
    }
}
