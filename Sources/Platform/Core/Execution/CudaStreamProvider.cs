using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace GoodAI.Core.Execution
{
    internal class CudaStreamProvider
    {
        public static CudaStreamProvider Instance { get; } = new CudaStreamProvider();

        public CudaStream CurrentStream { get; } = new CudaStream(CUstream.StreamPerThread);

        private readonly int StreamsPerThread = 8;

        public void SwitchToNextStream()
        {
            // TODO
        }
    }
}
