using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using YAXLib;

namespace GoodAI.Modules.Join
{
    /// <summary>
    /// 
    /// </summary>
    [Flags]
    public enum DistanceOperation
    {
        None = 0,

        DotProd = 1 << 1,
        CosDist = 1 << 2,

        EuclidDist = 1 << 5,
        EuclidDistSquared = 1 << 6,

        HammingDist = 1 << 7,
        HammingSim = 1 << 8, // hamming distance mapped to range <0,1>, where most similar vectors have similarity 1.
    }

    public class MyDistanceOps
    {
        private readonly MyWorkingNode m_caller;
        private DistanceOperation m_operations;
        private readonly MyMemoryBlock<float> m_temp;

        private readonly MyCudaKernel m_combineVecsKernel;
        private readonly MyProductKernel<float> m_dotKernel, m_cosKernel;
        private readonly MyReductionKernel<float> m_reduceSumKernel;


        public MyDistanceOps(MyWorkingNode caller, DistanceOperation operations, MyMemoryBlock<float> tempBlock = null)
        {
            m_caller = caller;
            m_operations = operations;
            m_temp = tempBlock;


            if (operations.HasFlag(DistanceOperation.DotProd))
            {
                m_dotKernel = MyKernelFactory.Instance.KernelProduct<float>(caller, caller.GPU, ProductMode.f_DotProduct_f);
            }

            if (operations.HasFlag(DistanceOperation.CosDist))
            {
                m_cosKernel = MyKernelFactory.Instance.KernelProduct<float>(caller, caller.GPU, ProductMode.f_Cosine_f);
            }

            if (operations.HasFlag(DistanceOperation.EuclidDist) || operations.HasFlag(DistanceOperation.EuclidDistSquared))
            {
                // EuclidDist computes EuclidDistSquared first, so keep them together:
                m_operations |= DistanceOperation.EuclidDist | DistanceOperation.EuclidDistSquared;
                m_dotKernel = MyKernelFactory.Instance.KernelProduct<float>(caller, caller.GPU, ProductMode.f_DotProduct_f);
            }

            if (operations.HasFlag(DistanceOperation.HammingDist))
            {
                m_reduceSumKernel = MyKernelFactory.Instance.KernelReduction<float>(caller, caller.GPU, ReductionMode.f_Sum_f);
            }
            if (operations.HasFlag(DistanceOperation.HammingSim))
            {
                m_reduceSumKernel = MyKernelFactory.Instance.KernelReduction<float>(caller, caller.GPU, ReductionMode.f_Sum_f);
            }

            if (operations.HasFlag(DistanceOperation.EuclidDist) || operations.HasFlag(DistanceOperation.EuclidDistSquared) || 
                operations.HasFlag(DistanceOperation.HammingDist) || operations.HasFlag(DistanceOperation.HammingSim))
            {
                m_combineVecsKernel = MyKernelFactory.Instance.Kernel(m_caller.GPU, @"Common\CombineVectorsKernel", "CombineTwoVectorsKernel");
            }
        }


        public void Run(DistanceOperation operation,
            CudaDeviceVariable<float> A, int sizeA,
            CudaDeviceVariable<float> B, int sizeB,
            CudaDeviceVariable<float> result, int sizeRes)
        {
            if (!ValidateAtRun(operation))
                return;

            switch (operation)
            {
                case DistanceOperation.DotProd:
                    //ZXC m_dotKernel.Run(result.DevicePointer, 0, A.DevicePointer, B.DevicePointer, sizeA, 0);
                    m_dotKernel.Run(result.DevicePointer, A.DevicePointer, B.DevicePointer, sizeA);
                    break;

                case DistanceOperation.CosDist:
                    //ZXC m_cosKernel.Run(result.DevicePointer, 0, A.DevicePointer, B.DevicePointer, sizeA, 0);
                    m_cosKernel.Run(result.DevicePointer, A.DevicePointer, B.DevicePointer, sizeA);
                    break;

                case DistanceOperation.EuclidDist:
                    float res = RunReturn(operation, A, sizeA, B, sizeB);
                    result.CopyToDevice(res);
                    break;

                case DistanceOperation.EuclidDistSquared:
                    m_combineVecsKernel.SetupExecution(sizeA);
                    m_combineVecsKernel.Run(A.DevicePointer, B.DevicePointer, m_temp, (int)MyJoin.MyJoinOperation.Subtraction, sizeA);
                    //ZXC m_dotKernel.Run(result.DevicePointer, 0, m_temp, m_temp, m_temp.Count, 0);
                    m_dotKernel.Run(result.DevicePointer, m_temp, m_temp);
                    break;

                case DistanceOperation.HammingDist:
                    m_combineVecsKernel.SetupExecution(sizeA);
                    m_combineVecsKernel.Run(A.DevicePointer, B.DevicePointer, m_temp, (int)MyJoin.MyJoinOperation.Equal, sizeA);
                    //ZXC m_reduceSumKernel.Run(result.DevicePointer, m_temp, m_temp.Count, 0, 0, 1, /*distributed = false*/0); // reduction to a single number
                    m_reduceSumKernel.Run(result.DevicePointer, m_temp);
                    float fDist = 0; // to transform number of matches to a number of differences
                    result.CopyToHost(ref fDist);
                    fDist = m_temp.Count - fDist;
                    result.CopyToDevice(fDist);
                    break;

                case DistanceOperation.HammingSim:
                    m_combineVecsKernel.SetupExecution(sizeA);
                    m_combineVecsKernel.Run(A.DevicePointer, B.DevicePointer, m_temp, (int)MyJoin.MyJoinOperation.Equal, sizeA);
                    //ZXC m_reduceSumKernel.Run(result.DevicePointer, m_temp, m_temp.Count, 0, 0, 1, /*distributed = false*/0); // reduction to a single number
                    m_reduceSumKernel.Run(result.DevicePointer, m_temp);
                    // take the single number (number of different bits) and convert it to Hamming Similarity: 
                    // a number in range <0,1> that says how much the vectors are similar
                    float fSim = 0;
                    result.CopyToHost(ref fSim);
                    fSim = fSim / m_temp.Count;
                    result.CopyToDevice(fSim);
                    break;
            }
        }

        public void Run(DistanceOperation operation, MyMemoryBlock<float> A, MyMemoryBlock<float> B, MyMemoryBlock<float> result)
        {
            Run(operation, A.GetDevice(m_caller), A.Count, B.GetDevice(m_caller), B.Count, result.GetDevice(m_caller), result.Count);
        }


        public float RunReturn(DistanceOperation operation,
            CudaDeviceVariable<float> A, int sizeA,
            CudaDeviceVariable<float> B, int sizeB)
        {
            if (m_temp == null)
            {
                MyLog.ERROR.WriteLine("Init the object with a valid temp block of size at least one to enable RunReturn.");
                return float.NaN;
            }

            switch (operation)
            {
                case DistanceOperation.None:
                    return float.NaN;

                case DistanceOperation.EuclidDist:
                    Run(DistanceOperation.EuclidDistSquared, A, sizeA, B, sizeB, m_temp.GetDevice(m_caller), sizeA);
                    m_temp.SafeCopyToHost(0, 1);

                    return (float)Math.Sqrt(m_temp.Host[0]);

                default:
                    Run(operation, A, sizeA, B, sizeB, m_temp.GetDevice(m_caller), sizeA);
                    m_temp.SafeCopyToHost(0, 1);

                    return m_temp.Host[0];
            }
        }

        public float RunReturn(DistanceOperation operation, MyMemoryBlock<float> A, MyMemoryBlock<float> B)
        {
            return RunReturn(operation, A.GetDevice(m_caller), A.Count, B.GetDevice(m_caller), B.Count);
        }

        // can be called during simulation
        public bool ValidateAtRun(DistanceOperation operation)
        {
            if (operation == DistanceOperation.None)
                return false;

            if ((operation & m_operations) == 0)
            {
                MyLog.WARNING.WriteLine("Trying to execute an uninitialized distance operation. Owner: " + m_caller.Name);
                return false;
            }

            return true;
        }

        // should be called before simulation is started
        public static bool Validate(DistanceOperation operation, int sizeA, int sizeB, int sizeTemp, int sizeRes, out string errorOutput)
        {
            errorOutput = null;

            switch (operation)
            {
                case DistanceOperation.None:
                    return true;

                case DistanceOperation.DotProd:
                case DistanceOperation.CosDist:
                    break;

                case DistanceOperation.EuclidDist:
                case DistanceOperation.EuclidDistSquared:
                case DistanceOperation.HammingDist:
                case DistanceOperation.HammingSim:
                    if (sizeA != sizeTemp)
                    {
                        errorOutput = "Invalid temp block size for the distance operation.";
                        return false;
                    }
                    break;

                default:
                    errorOutput = "Invalid operation. Only a single value within the enum range should be passed.";
                    return false;
            }

            if (sizeA != sizeB || sizeRes != 1)
            {
                errorOutput = "Invalid input/output block sizes for the distance operation.";
                return false;
            }

            return true;
        }
    }
}
