using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Group;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.RBM
{
    /// <author>GoodAI</author>
    /// <meta>mz</meta>
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

        [YAXSerializableField(DefaultValue = 0.0f)]
        [MyBrowsable, Category("\tRegularization"), Description("Only used for backprop in neural network, not used in RBM mode!"), DisplayName("Backprop dropout")]
        public new float Dropout { get; set; }
    }
    

}
