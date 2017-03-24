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

        public const int NoLayer = -1;

        private int m_lastLayerNumber = NoLayer;
        private int m_callsWithinLayer;

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
            if (layerNumber < 0)
            {
                if ((layerNumber == NoLayer) && (m_lastLayerNumber != NoLayer))
                {
                    MyLog.INFO.WriteLine($"{nameof(CudaSyncHelper)}.{nameof(OnStartExecute)}: "
                                         + "Called with default layer even if explicit layer is used in other cases.");
                }

                return;
            }

            SyncFirstInLayer(layerNumber);

            m_streamProvider.SwitchToNextStream();
        }

        public void SwitchToNextStream() => m_streamProvider.SwitchToNextStream();

        private void SyncFirstInLayer(int layerNumber)
        {
            if (layerNumber < m_lastLayerNumber)
            {
                MyLog.DEBUG.WriteLine(
                    $"{nameof(CudaSyncHelper)}: Layer number {layerNumber} is smaller than the last one ({m_lastLayerNumber})."
                    + " Assuming new simulation step.");

                m_lastLayerNumber = -1;
            }

            if ((m_lastLayerNumber > -1) && (layerNumber >= m_lastLayerNumber + 2))
                MyLog.WARNING.WriteLine($"{nameof(SyncFirstInLayer)}: Layer number jumped by {layerNumber - m_lastLayerNumber}." 
                    + " Skipped some layers?");

            if (layerNumber > m_lastLayerNumber)
            {
                MyLog.DEBUG.WriteLine($"{nameof(CudaSyncHelper)}: Synchronizing at the start of a new layer ({layerNumber}).");

                m_streamProvider.SynchronizeAllStreams();

                m_callsWithinLayer = 0;
            }

            m_callsWithinLayer++;
            if (m_callsWithinLayer > 1000)
            {
                MyLog.WARNING.WriteLine($"{nameof(CudaSyncHelper)}: There has been over thousand calls from one layer."
                                        + " You need at least two layers to enable synchronization.");
                m_callsWithinLayer = 0;  // Only say this from time to time.
            }


            m_lastLayerNumber = layerNumber;
        }
    }
}