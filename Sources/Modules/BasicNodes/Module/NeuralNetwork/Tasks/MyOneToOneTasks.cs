using GoodAI.Core;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Modules.NeuralNetwork.Tasks
{
    [Description("FeedForward"), MyTaskInfo(OneShot = false)]
    public class MyOneToOneForwardTask : MyAbstractForwardTask<MyAbstractLayer>
    {
        public MyOneToOneForwardTask() { } //parameterless constructor

        private MyCudaKernel m_forwardKernel; // kernel
        private MyCudaKernel m_softmaxKernel;
        public override void Init(int nGPU)
        {
            m_forwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\FeedForwardKernels", "OneToOneForwardKernel");
            m_softmaxKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Activation\ActivationFunction", "SoftmaxKernel");
        }

        public override void Execute() //Task execution
        {
            // do a trick to avoid infinity (NaN) problems with exponential values in softmax
            if (Owner.ActivationFunction == ActivationFunctionType.SOFTMAX)
            {
                Owner.Output.SafeCopyToHost();
                float[] f = Owner.Output.Host;

                float max = f.Max();
                for (int i = 0; i < f.Length; i++)
                {
                    f[i] -= max;
                }

                Array.Copy(f, Owner.Output.Host, f.Length);

                Owner.Output.SafeCopyToDevice();
            }

            m_forwardKernel.SetupExecution(Owner.Neurons);
            m_forwardKernel.Run(
                (int)Owner.ActivationFunction,
                Owner.Input,
                Owner.Output,
                Owner.Neurons
                );

            if (Owner.ActivationFunction == ActivationFunctionType.SOFTMAX)
            {
                Owner.Output.SafeCopyToHost();

                float expSum = Owner.Output.Host.Sum();

                m_softmaxKernel.SetupExecution(Owner.Neurons);
                m_softmaxKernel.Run(
                    Owner.Output,
                    expSum,
                    Owner.Neurons
                    );

            }
        }
    }

    [Description("DeltaBack"), MyTaskInfo(OneShot = false)]
    public class MyOneToOneDeltaBackTask : MyAbstractBackDeltaTask<MyAbstractLayer>
    {
        public MyOneToOneDeltaBackTask() { } //parameterless constructor

        private MyCudaKernel m_deltaKernel; // kernel
        public override void Init(int nGPU)
        {
            m_deltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\DeltaKernels", "OneToOneDeltaKernel");
        }

        public override void Execute() //Task execution
        {
            // pointer to previous layer
            MyAbstractLayer previousLayer = Owner.PreviousLayer;

            if (previousLayer != null)
            {
                //// reset delta
                //previousLayer.Delta.Fill(0); // do this after updating weights (batch learning)

                // determine input to previous layer
                CUdeviceptr prevInputPtr;
                if (previousLayer is MyAbstractWeightLayer)
                    prevInputPtr = (previousLayer as MyAbstractWeightLayer).NeuronInput.GetDevicePtr(previousLayer.GPU);
                else
                    prevInputPtr = previousLayer.Input.GetDevicePtr(previousLayer.GPU);

                m_deltaKernel.SetupExecution(Owner.Neurons);
                m_deltaKernel.Run(
                    (int)previousLayer.ActivationFunction,
                    prevInputPtr,
                    previousLayer.Delta,
                    Owner.Delta,
                    Owner.Neurons
                    );
            }
        }
    }
}
