using System;
using System.IO;
using System.Linq;
using GoodAI.Core;
using GoodAI.Core.Memory;
using ManagedCuda;

namespace GoodAI.Modules.Transforms
{
    public static class MyReductionFactory
    {
        public enum Mode
        {
            i_Sum_i, i_MinIdx_2i, i_MaxIdx_2i, i_MinIdxMaxIdx_4i,
            f_Sum_f, f_MinIdx_fi, f_MinIdx_ff, f_MaxIdx_fi, f_MaxIdx_ff, f_MinIdxMaxIdx_fifi, f_MinMax_2f,
            f_DotProduct_f, i_DotProduct_i, f_Cosine_f, c_ComplexDot_c
        };

        public static MyCudaKernel Kernel(int nGPU, Mode mode, int blockCount = 1, bool segmented = false)
        {
            uint outSize = 0;
            MyCudaKernel kernel = null;

            if (blockCount < 1)
                throw new ArgumentOutOfRangeException("blockCount", "Invalid block count -- must be positive.");

            if (!segmented)
            {
                switch (mode)
                {
                    case Mode.i_Sum_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI7i_Sum_iiLj512EEvPvPVKvjjjjb"); outSize = 4; break;
                    case Mode.i_MinIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11i_MinIdx_2iiLj512EEvPvPVKvjjjjb"); outSize = 8; break;
                    case Mode.i_MaxIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11i_MaxIdx_2iiLj512EEvPvPVKvjjjjb"); outSize = 8; break;
                    case Mode.i_MinIdxMaxIdx_4i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI17i_MinIdxMaxIdx_4iiLj512EEvPvPVKvjjjjb"); outSize = 16; break;
                    case Mode.f_Sum_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI7f_Sum_ffLj512EEvPvPVKvjjjjb"); outSize = 4; break;
                    case Mode.f_MinIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MinIdx_fifLj512EEvPvPVKvjjjjb"); outSize = 8; break;
                    case Mode.f_MinIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MinIdx_fffLj512EEvPvPVKvjjjjb"); outSize = 8; break;
                    case Mode.f_MaxIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MaxIdx_fifLj512EEvPvPVKvjjjjb"); outSize = 8; break;
                    case Mode.f_MaxIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MaxIdx_fffLj512EEvPvPVKvjjjjb"); outSize = 8; break;
                    case Mode.f_MinMax_2f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MinMax_2ffLj512EEvPvPVKvjjjjb"); outSize = 8; break;
                    case Mode.f_MinIdxMaxIdx_fifi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI19f_MinIdxMaxIdx_fififLj512EEvPvPVKvjjjjb"); outSize = 16; break;
                    case Mode.i_DotProduct_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI7i_Dot_iiLj512EEvPvjPVKvS3_jb"); outSize = 4; break;
                    case Mode.f_DotProduct_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI7f_Dot_ffLj512EEvPvjPVKvS3_jb"); outSize = 4; break;
                    case Mode.f_Cosine_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI10f_Cosine_ffLj512EEvPvjPVKvS3_jb"); outSize = 16; break;
                    case Mode.c_ComplexDot_c: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI14c_ComplexDot_c7ComplexLj512EEvPvjPVKvS4_jb"); outSize = 8; break;
                    default: throw new ArgumentOutOfRangeException("mode", "Unrecognized reduction mode.");
                }
            }
            else
            {
                switch (mode)
                {
                    case Mode.i_Sum_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI7i_Sum_iiLj512EEvPvPVKvjj"); outSize = 4; break;
                    case Mode.i_MinIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11i_MinIdx_2iiLj512EEvPvPVKvjj"); outSize = 8; break;
                    case Mode.i_MaxIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11i_MaxIdx_2iiLj512EEvPvPVKvjj"); outSize = 8; break;
                    case Mode.i_MinIdxMaxIdx_4i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI17i_MinIdxMaxIdx_4iiLj512EEvPvPVKvjj"); outSize = 16; break;
                    case Mode.f_Sum_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI7f_Sum_ffLj512EEvPvPVKvjj"); outSize = 4; break;
                    case Mode.f_MinIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MinIdx_fifLj512EEvPvPVKvjj"); outSize = 8; break;
                    case Mode.f_MinIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MinIdx_fffLj512EEvPvPVKvjj"); outSize = 8; break;
                    case Mode.f_MaxIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MaxIdx_fifLj512EEvPvPVKvjj"); outSize = 8; break;
                    case Mode.f_MaxIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MaxIdx_fffLj512EEvPvPVKvjj"); outSize = 8; break;
                    case Mode.f_MinMax_2f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MinMax_2ffLj512EEvPvPVKvjj"); outSize = 8; break;
                    case Mode.f_MinIdxMaxIdx_fifi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI19f_MinIdxMaxIdx_fififLj512EEvPvPVKvjj"); outSize = 16; break;
                    case Mode.i_DotProduct_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z11SDotProductI7i_Dot_iiLj512EEvPvPVKvS3_j"); outSize = 4; break;
                    case Mode.f_DotProduct_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z11SDotProductI7f_Dot_ffLj512EEvPvPVKvS3_j"); outSize = 4; break;
                    case Mode.f_Cosine_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z11SDotProductI10f_Cosine_ffLj512EEvPvPVKvS3_j"); outSize = 16; break;
                    case Mode.c_ComplexDot_c: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z11SDotProductI14c_ComplexDot_c7ComplexLj512EEvPvPVKvS4_j"); outSize = 8; break;
                    default: throw new ArgumentOutOfRangeException("mode", "Unrecognized reduction mode.");
                }
            }

            kernel.DynamicSharedMemory = 512 * outSize;
            kernel.GridDimensions = blockCount;
            kernel.BlockDimensions = 512;
            return kernel;
        }

        public static MyCudaKernel AsyncKernel(int nGPU, Mode mode, int tempBlockSize, int segmentCount = 1, bool segmented = false)
        {
            uint outSize = 0;
            MyCudaKernel kernel = null;

            if (segmentCount < 1)
                throw new ArgumentOutOfRangeException("segmentCount", "Invalid block count -- must be positive.");
            if (!segmented && segmentCount > 1) // The global mem barrier in reduction.cu will may create race conditions when called concurrently
                throw new ArgumentOutOfRangeException("segmentCount", "Non-segmented mode called with multiple blocks is not allowed.");

            if (!segmented)
            {
                switch (mode)
                {
                    case Mode.i_Sum_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI7i_Sum_iiLj512EEvPvPVKvS1_jjjjb"); outSize = 4; break;
                    case Mode.i_MinIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11i_MinIdx_2iiLj512EEvPvPVKvS1_jjjjb"); outSize = 8; break;
                    case Mode.i_MaxIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11i_MaxIdx_2iiLj512EEvPvPVKvS1_jjjjb"); outSize = 8; break;
                    case Mode.i_MinIdxMaxIdx_4i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI17i_MinIdxMaxIdx_4iiLj512EEvPvPVKvS1_jjjjb"); outSize = 16; break;
                    case Mode.f_Sum_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI7f_Sum_ffLj512EEvPvPVKvS1_jjjjb"); outSize = 4; break;
                    case Mode.f_MinIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MinIdx_fifLj512EEvPvPVKvS1_jjjjb"); outSize = 8; break;
                    case Mode.f_MinIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MinIdx_fffLj512EEvPvPVKvS1_jjjjb"); outSize = 8; break;
                    case Mode.f_MaxIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MaxIdx_fifLj512EEvPvPVKvS1_jjjjb"); outSize = 8; break;
                    case Mode.f_MaxIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MaxIdx_fffLj512EEvPvPVKvS1_jjjjb"); outSize = 8; break;
                    case Mode.f_MinMax_2f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MinMax_2ffLj512EEvPvPVKvS1_jjjjb"); outSize = 8; break;
                    case Mode.f_MinIdxMaxIdx_fifi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI19f_MinIdxMaxIdx_fififLj512EEvPvPVKvS1_jjjjb"); outSize = 16; break;
                    case Mode.i_DotProduct_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI7i_Dot_iiLj512EEvPvjPVKvS3_S1_jb"); outSize = 4; break;
                    case Mode.f_DotProduct_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI7f_Dot_ffLj512EEvPvjPVKvS3_S1_jb"); outSize = 4; break;
                    case Mode.f_Cosine_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI10f_Cosine_ffLj512EEvPvjPVKvS3_S1_jb"); outSize = 16; break;
                    case Mode.c_ComplexDot_c: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI14c_ComplexDot_c7ComplexLj512EEvPvjPVKvS4_jb"); outSize = 8; break;
                    default: throw new ArgumentOutOfRangeException("mode", "Unrecognized reduction mode.");
                }
            }
            else
            {
                switch (mode)
                {
                    case Mode.i_Sum_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI7i_Sum_iiLj512EEvPvPVKvS1_jj"); outSize = 4; break;
                    case Mode.i_MinIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11i_MinIdx_2iiLj512EEvPvPVKvS1_jj"); outSize = 8; break;
                    case Mode.i_MaxIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11i_MaxIdx_2iiLj512EEvPvPVKvS1_jj"); outSize = 8; break;
                    case Mode.i_MinIdxMaxIdx_4i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI17i_MinIdxMaxIdx_4iiLj512EEvPvPVKvS1_jj"); outSize = 16; break;
                    case Mode.f_Sum_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI7f_Sum_ffLj512EEvPvPVKvS1_jj"); outSize = 4; break;
                    case Mode.f_MinIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MinIdx_fifLj512EEvPvPVKvS1_jj"); outSize = 8; break;
                    case Mode.f_MinIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MinIdx_fffLj512EEvPvPVKvS1_jj"); outSize = 8; break;
                    case Mode.f_MaxIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MaxIdx_fifLj512EEvPvPVKvS1_jj"); outSize = 8; break;
                    case Mode.f_MaxIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MaxIdx_fffLj512EEvPvPVKvS1_jj"); outSize = 8; break;
                    case Mode.f_MinMax_2f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MinMax_2ffLj512EEvPvPVKvS1_jj"); outSize = 8; break;
                    case Mode.f_MinIdxMaxIdx_fifi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI19f_MinIdxMaxIdx_fififLj512EEvPvPVKvS1_jj"); outSize = 16; break;
                    case Mode.i_DotProduct_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z11SDotProductI7i_Dot_iiLj512EEvPvPVKvS3_S1_j"); outSize = 4; break;
                    case Mode.f_DotProduct_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z11SDotProductI7f_Dot_ffLj512EEvPvPVKvS3_S1_j"); outSize = 4; break;
                    case Mode.f_Cosine_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z11SDotProductI10f_Cosine_ffLj512EEvPvPVKvS3_S1_j"); outSize = 16; break;
                    case Mode.c_ComplexDot_c: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z11SDotProductI14c_ComplexDot_c7ComplexLj512EEvPvPVKvS4_j"); outSize = 8; break;
                    default: throw new ArgumentOutOfRangeException("mode", "Unrecognized reduction mode.");
                }
            }

            kernel.DynamicSharedMemory = 512 * outSize;
            kernel.GridDimensions = segmentCount;
            kernel.BlockDimensions = 512;

            if (tempBlockSize < kernel.GridDimensions.x * kernel.GridDimensions.y * kernel.GridDimensions.z * 2)
                throw new ArgumentOutOfRangeException("tempBlockSize", "Temp block size is too small to use with this setting.");

            return kernel;
        }
    }
}
