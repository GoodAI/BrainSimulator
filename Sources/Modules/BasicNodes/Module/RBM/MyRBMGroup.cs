using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Modules.RBM
{
    /// <author>Mikuláš Zelinka</author>
    /// <status>Working</status>
    /// <summary>
    ///     Restricted Boltzmann Machine node group.
    /// </summary>
    /// <description>
    ///     Node group used for Restricted Boltzmann Machines and deep learning.
    ///     Derived from Neural Network group whose functionality it inherits.
    ///     Can be used for initialization of weights prior to using SGD.
    /// </description>
    public class MyRBMGroup : MyNeuralNetworkGroup
    {
        [MyTaskGroup("BackPropagation")]
        public MyRBMLearningTask RBMLearning { get; private set; }
        [MyTaskGroup("BackPropagation")]
        public MyRBMReconstructionTask RBMReconstruction { get; private set; }
    }
    

}
