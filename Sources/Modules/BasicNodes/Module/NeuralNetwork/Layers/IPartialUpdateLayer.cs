using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Memory;

namespace GoodAI.Modules.NeuralNetwork.Layers
{
    // For layers that want to propagate deltas without changing all their weights
    public interface IPartialUpdateLayer
    {
        //public MyMemoryBlock<float> GetUpdateMask();

        // Index of first neuron for which to suppress weight updates
        int SuppressUpdatesAt();

        // The number of neurons for which to suppress weight updates
        int SuppressUpdatesCount();
    }
}
