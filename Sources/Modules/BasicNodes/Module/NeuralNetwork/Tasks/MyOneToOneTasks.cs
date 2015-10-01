using GoodAI.Core;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Layers;
using ManagedCuda.BasicTypes;
using System;
using System.ComponentModel;
using System.Linq;

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

            m_forwardKernel.SetupExecution(Owner.Neurons);
            m_forwardKernel.Run(
                (int)Owner.ActivationFunction,
                Owner.Input,
                Owner.Output,
                Owner.Neurons
                );

            // do a trick to avoid infinity (NaN) problems with exponential values in softmax
            if (Owner.ActivationFunction == ActivationFunctionType.SOFTMAX)
            {
                Owner.Output.SafeCopyToHost();

                float expSum = 0;

                float[] f = Owner.Output.Host;
                float max = f.Max();

                for (int i = 0; i < f.Length; i++)
                {
                    float exp = (float) Math.Exp(f[i] - max);
                    f[i] = exp;
                    expSum += exp;
                }


                // CPU version of the commented kernel below
                for (int i = 0; i < f.Length; i++)
                {
                    f[i] /= expSum;
                }

                Array.Copy(f, Owner.Output.Host, f.Length);

                Owner.Output.SafeCopyToDevice();

                /* 
                 * GPU version is slower, don't use it for now
                m_softmaxKernel.SetupExecution(Owner.Neurons);
                m_softmaxKernel.Run(
                    Owner.Output,
                    expSum,
                    Owner.Neurons
                    );*/

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
            foreach (MyConnection connection in Owner.InputConnections)
            {
                if (connection != null && connection.From is MyAbstractLayer)
                {
                    MyAbstractLayer prevLayer = connection.From as MyAbstractLayer;

                    // reset delta
                    prevLayer.Delta.Fill(0);

                    // determine input to previous layer
                    CUdeviceptr prevInputPtr = MyAbstractLayer.DetermineInput(prevLayer);

                    m_deltaKernel.SetupExecution(Owner.Neurons);
                    m_deltaKernel.Run(
                        (int)prevLayer.ActivationFunction,
                        prevInputPtr,
                        prevLayer.Delta,
                        Owner.Delta,
                        Owner.Neurons
                    );
                }
            }
        }
    }
}
