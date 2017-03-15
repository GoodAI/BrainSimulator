using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;

namespace GoodAI.Core.Execution
{
    internal class CudaStreamHandle
    {
        // TODO: with multi-threading the stream will need to be injected and stored here (another option is a thread-local storage)
        public CudaStream CurrentStream => m_streamProvider.CurrentStream;

        private readonly CudaStreamProvider m_streamProvider;

        public CudaStreamHandle(CudaStreamProvider source)
        {
            m_streamProvider = source;
        }
    }
}
