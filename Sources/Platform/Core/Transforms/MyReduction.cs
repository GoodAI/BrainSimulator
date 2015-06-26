using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BrainSimulator.Transforms
{
    public static class MyReductionFactory
    {
        public enum Mode
        {
            i_Sum_i, i_MinIdx_2i, i_MaxIdx_2i, i_MinIdxMaxIdx_4i,
            f_Sum_f, f_MinIdx_fi, f_MinIdx_ff, f_MaxIdx_fi, f_MaxIdx_ff, f_MinIdxMaxIdx_fifi, f_MinMax_2f,
            f_DotProduct_f, i_DotProduct_i, f_Cosine_f
        };

        public static MyCudaKernel Kernel(int nGPU, Mode mode)
        {
            uint outSize = 0;
            MyCudaKernel kernel = null;
            switch (mode)
            {
                case Mode.i_Sum_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Reduction/Reduction", "_Z9ReductionI7i_Sum_iiLj1024EEvPvPVKvjjjj"); outSize = 4; break;
                case Mode.i_MinIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Reduction/Reduction", "_Z9ReductionI11i_MinIdx_2iiLj1024EEvPvPVKvjjjj"); outSize = 8; break;
                case Mode.i_MaxIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Reduction/Reduction", "_Z9ReductionI11i_MaxIdx_2iiLj1024EEvPvPVKvjjjj"); outSize = 8; break;
                case Mode.i_MinIdxMaxIdx_4i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Reduction/Reduction", "_Z9ReductionI17i_MinIdxMaxIdx_4iiLj1024EEvPvPVKvjjjj"); outSize = 16; break;
                case Mode.f_Sum_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Reduction/Reduction", "_Z9ReductionI7f_Sum_ffLj1024EEvPvPVKvjjjj"); outSize = 4; break;
                case Mode.f_MinIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Reduction/Reduction", "_Z9ReductionI11f_MinIdx_fifLj1024EEvPvPVKvjjjj"); outSize = 8; break;
                case Mode.f_MinIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Reduction/Reduction", "_Z9ReductionI11f_MinIdx_fffLj1024EEvPvPVKvjjjj"); outSize = 8; break;
                case Mode.f_MaxIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Reduction/Reduction", "_Z9ReductionI11f_MaxIdx_fifLj1024EEvPvPVKvjjjj"); outSize = 8; break;
                case Mode.f_MaxIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Reduction/Reduction", "_Z9ReductionI11f_MaxIdx_fffLj1024EEvPvPVKvjjjj"); outSize = 8; break;
                case Mode.f_MinMax_2f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Reduction/Reduction", "_Z9ReductionI11f_MinMax_2ffLj1024EEvPvPVKvjjjj"); outSize = 8; break;
                case Mode.f_MinIdxMaxIdx_fifi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Reduction/Reduction", "_Z9ReductionI19f_MinIdxMaxIdx_fififLj1024EEvPvPVKvjjjj"); outSize = 16; break;
                case Mode.i_DotProduct_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Reduction/Reduction", "_Z10DotProductI7i_Dot_iiLj1024EEvPvjPVKvS3_j"); outSize = 4; break;
                case Mode.f_DotProduct_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Reduction/Reduction", "_Z10DotProductI7f_Dot_ffLj1024EEvPvjPVKvS3_j"); outSize = 4; break;
                case Mode.f_Cosine_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Reduction/Reduction", "_Z10DotProductI10f_Cosine_ffLj1024EEvPvjPVKvS3_j"); outSize = 16; break;
            }
            kernel.DynamicSharedMemory = 1024 * outSize;
            kernel.BlockDimensions = 1024;
            kernel.GridDimensions = 10;
            return kernel;
        }

    }
}
