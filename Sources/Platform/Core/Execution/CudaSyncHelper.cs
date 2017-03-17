using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Utils;

namespace GoodAI.Core.Execution
{
    /// <summary>
    /// Experimental! (Used in a hackish implementation of using multiple CUDA streams from one CPU thread.)
    /// </summary>
    public sealed class CudaSyncHelper
    {
        public static CudaSyncHelper Instance { get; } = new CudaSyncHelper(null);

        private int m_lastLayerNumber = -1;
        private int m_topLayerNumber = -1;

        private readonly CudaStreamProvider m_streamProvider;

        internal CudaSyncHelper(CudaStreamProvider streamProvider)
        {
            m_streamProvider = streamProvider ?? CudaStreamProvider.Instance;
        }

        /// <summary>
        /// Experimental! Call synchronize all streams for the first node in each layer. Switches to next stream every time.
        /// </summary>
        /// <param name="layerNumber">Layer number of the calling node.</param>
        public void OnStartExecute(int layerNumber)
        {
            SyncFirstInLayer(layerNumber);

            m_streamProvider.SwitchToNextStream();
        }

        private void SyncFirstInLayer(int layerNumber)
        {
            if (layerNumber < m_lastLayerNumber)
            {
                MyLog.DEBUG.WriteLine(
                    $"{nameof(CudaSyncHelper)}: Layer number {layerNumber} is smaller than the last one ({m_lastLayerNumber})."
                    + " Assuming new simulation step.");

                m_topLayerNumber = m_lastLayerNumber;
                m_lastLayerNumber = -1;
            }

            if ((m_lastLayerNumber > -1) && (layerNumber >= m_lastLayerNumber + 2))
                MyLog.WARNING.WriteLine($"{nameof(SyncFirstInLayer)}: Layer number jumped by {layerNumber - m_lastLayerNumber}." 
                    + " Skipped some layers?");

            if (layerNumber > m_lastLayerNumber)
            {
                MyLog.DEBUG.WriteLine($"{nameof(CudaSyncHelper)}: Synchronizing at the start of a new layer ({layerNumber}).");
                m_streamProvider.SynchronizeAllStreams();
            }

            m_lastLayerNumber = layerNumber;
        }
    }
}