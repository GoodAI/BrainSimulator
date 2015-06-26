using BrainSimulator.NeuralNetwork.Group;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace BrainSimulator.RBM
{
    /// <author>Mikulas Zelinka</author>
    /// <status>Working</status>
    /// <summary>
    ///     Node group used for Restricted Boltzmann Machines and deep learning.
    ///     Derived from Neural Network group whose functionality it inherits.
    /// </summary>
    /// <description>
    ///     Node group used for Restricted Boltzmann Machines and deep learning.
    ///     Derived from Neural Network group whose functionality it inherits.
    /// </description>
    public class MyRBMGroup : MyNeuralNetworkGroup
    {
        [MyTaskGroup("BackPropagation")]
        public MyRBMLearningTask RBMLearning { get; private set; }
        [MyTaskGroup("BackPropagation")]
        public MyRBMReconstructionTask RBMReconstruction { get; private set; }
    }
    

}
