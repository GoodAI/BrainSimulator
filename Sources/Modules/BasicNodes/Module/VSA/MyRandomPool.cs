using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using System;
using System.ComponentModel;
using System.Diagnostics;
using YAXLib;

namespace GoodAI.Modules.VSA
{
    public abstract class MyRandomPool : MyWorkingNode
    {
        public enum AxisToNormalizeEnum
        {
            yDim,
            xDim,
        }

        public enum VectorGenerationMode
        {
            Normal,
            Orthonormalize,
            AverageBaseVectors,
        }


        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }


        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 1024)]
        [Description("The length of the symbolic vectors used.")]
        public int SymbolSize { get; set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 32)]
        [Description("Specifies the desired output block's column hint.")]
        public int ColumnHint { get; set; }

        private static bool _useBSCVariety = false;

        [MyBrowsable, Category("SHARED: Binary Spatter Code")]
        [YAXSerializableField(DefaultValue = false)]
        [Description("Generate binary values instead of floats.")]
        public bool UseBSCVariety
        {
            get { return _useBSCVariety; }
            set { _useBSCVariety = value; }
        }

        private static float _oneToZeroRatio = 1f;

        [MyBrowsable, Category("SHARED: Binary Spatter Code")]
        [YAXSerializableField(DefaultValue = 1f)]
        public float OneToZeroRatio
        {
            get { return _oneToZeroRatio; }
            set { _oneToZeroRatio = Math.Max(0, Math.Min(1, value)); }
        }

        [MyBrowsable, Category("Value generation"),
        Description("Specifies the transformation matrix generation mode." +
                    "Orthonormalization will make the vectors along the longer axis orthonormal." +
                    "AverageBaseVectors will compute ALL the bases of the larger dimension and will " +
                    "evenly average them to get the correct dimension.")]
        [YAXSerializableField(DefaultValue = VectorGenerationMode.Normal)]
        public VectorGenerationMode VectorMode { get; set; }


        [MyBrowsable, Category("Function")]
        public abstract int Seed { get; }

        protected abstract string GlobalVariableName { get; }
        protected abstract int PatternCount { get; }


        protected virtual float[] GenerateRandomVectors()
        {
            float[] codeVectors = new float[SymbolSize * PatternCount];
            Random random = new Random(Seed);

            if (UseBSCVariety)
            {
                GenerateRandomBSCVectors(codeVectors, random, OneToZeroRatio);
            }
            else
            {
                Random oneToZeroRandom = new Random((int)((ulong)Seed * 2147483647) % 524287);
                GenerateRandomNormalVectors(codeVectors, random, SymbolSize, PatternCount,
                    normalize: VectorMode != VectorGenerationMode.Orthonormalize,
                    oneToZeroRandom: oneToZeroRandom,
                    oneToZeroRatio: OneToZeroRatio);

                if (VectorMode == VectorGenerationMode.Orthonormalize)
                    OrthonormalizeVectors(codeVectors, SymbolSize, PatternCount);
            }

            return codeVectors;
        }


        #region Static methods

        /// <summary>
        /// Fills the <paramref name="codeVectors"/> array with <paramref name="otherDim"/> random normally distributed vectors with size <paramref name="leadingDim"/>, zero mean and variance 1/<paramref name="var"/>.
        /// Each vector is normalized to unit length. Use the <paramref name="oneToZeroRatio"/> parameter to specify the sparseness of the vectors.
        /// <paramref name="otherDim"/> shall be the leading dimension of the resulting matrix.
        /// </summary>
        /// <param name="codeVectors">The array to populate by the random normal vectors.</param>
        /// <param name="random">The random object used to generate vector elements.</param>
        /// <param name="leadingDim">The size of each vector.</param>
        /// <param name="otherDim">The number of vectors to be generated.</param>
        /// <param name="mean">The expected mean of the generated values. Defaults to zero.</param>
        /// <param name="var">The variance of the generated values. If not specified or non-positive variance is passed, the value 1/xDim will be used to generate approximately normalized vectors.</param>
        /// <param name="normalize">Specifies whether each generated vector should be normalized to unit length.</param>
        /// <param name="oneToZeroRandom">The random object used to decide the sparseness of the vectors. If null is passed, <paramref name="oneToZeroRatio"/> is set to 1.</param>
        /// <param name="oneToZeroRatio">The ratio of non-zero elements to zeros.</param>
        public static void GenerateRandomNormalVectors(float[] codeVectors, Random random, int leadingDim, int otherDim, float mean = 0, double var = 0, bool normalize = true, Random oneToZeroRandom = null, float oneToZeroRatio = 1)
        {
            Debug.Assert(leadingDim > 0 && otherDim > 0, "Negative matrix dimensions");
            Debug.Assert(codeVectors != null && codeVectors.Length >= leadingDim * otherDim, "Invalid codeVectors length");
            Debug.Assert(random != null, "Missing random object");

            const float smallConstant = 0.0000001f;


            if (var < smallConstant)
                var = 1d / leadingDim;

            float stdDevRoot = (float)(Math.Sqrt(var));

            if (oneToZeroRatio > smallConstant)  // The vector is normalized in the end
                stdDevRoot /= oneToZeroRatio;

            for (int i = 0; i < leadingDim * otherDim; i += leadingDim)
            {
                float sum = 0;

                if (oneToZeroRandom != null && oneToZeroRatio < 1)
                {
                    for (int j = i; j < i + leadingDim; j++)
                    {
                        if (oneToZeroRandom.NextDouble() <= oneToZeroRatio)
                        {
                            double u1 = random.NextDouble();
                            double u2 = random.NextDouble();
                            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                            //random normal(0,1)

                            codeVectors[j] = mean + stdDevRoot * (float)randStdNormal;
                            sum += codeVectors[j] * codeVectors[j];
                        }
                        else
                            codeVectors[j] = 0;
                    }
                }
                else
                {
                    for (int j = i; j < i + leadingDim; j++)
                    {
                        double u1 = random.NextDouble();
                        double u2 = random.NextDouble();
                        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                        //random normal(0,1)

                        codeVectors[j] = mean + stdDevRoot * (float)randStdNormal;
                        sum += codeVectors[j] * codeVectors[j];
                    }
                }

                if (!normalize)
                    continue;

                if (sum < smallConstant) // do while?
                    continue;

                sum = 1 / (float)Math.Sqrt(sum);

                for (int j = i; j < i + leadingDim; j++)
                    codeVectors[j] *= sum;
            }
        }

        /// <summary>
        /// Fills the <paramref name="codeVectors"/> array with random binary numbers.
        /// Use the <paramref name="oneToZeroRatio"/> parameter to specify the sparseness of the vectors.
        /// </summary>
        /// <param name="codeVectors">The array to populate by the random binary vectors.</param>
        /// <param name="random">The random object used to decide the sparseness of the vectors. If null is passed, <paramref name="oneToZeroRatio"/> is set to 1.</param>
        /// <param name="oneToZeroRatio">The ratio of non-zero elements to zeros.</param>
        public static void GenerateRandomBSCVectors(float[] codeVectors, Random random, float oneToZeroRatio = 1)
        {
            Debug.Assert(codeVectors != null, "Invalid codeVectors length");
            Debug.Assert(random != null, "Missing random object");


            for (int i = 0; i < codeVectors.Length; i++)
                codeVectors[i] = random.NextDouble() <= oneToZeroRatio ? 1 : 0;
        }


        /// <summary>
        /// Transposes the matrix with dimensions specified by <paramref name="xDim"/> and <paramref name="yDim"/>.
        /// </summary>
        public static void Transpose(ref float[] buffer, int xDim, int yDim)
        {
            Debug.Assert(buffer != null, "Missing buffer to transpose.");


            float[] tmp = new float[buffer.Length];

            int idx = -1;

            for (int i = 0; i < xDim; i++)
                for (int j = 0; j < yDim; j++)
                    tmp[j * xDim + i] = buffer[++idx];

            buffer = tmp;
        }

        /// <summary>
        /// Normalizes vectors along the leading dimension.
        /// </summary>
        public static void NormalizeLeadingDim(float[] buffer, int leadingDim, int otherDim)
        {
            Debug.Assert(buffer != null, "Missing buffer to normalize.");


            for (int i = 0; i < otherDim; i++)
            {
                var seg = new ArraySegment<float>(buffer, i * leadingDim, leadingDim);
                double lenSq = DotProduct(seg, seg);
                float factor = (float)(1 / Math.Sqrt(lenSq));

                for (int j = seg.Offset; j < seg.Offset + seg.Count; j++)
                    seg.Array[j] *= factor;
            }
        }

        /// <summary>
        /// Normalizes vectors along the leading dimension.
        /// </summary>
        public static void NormalizeLeadingDim(
            MyMemoryBlock<float> vectors, MyMemoryBlock<float> temp,
            int leadingDim, int otherDim,
            MyProductKernel<float> dotKernel, MyCudaKernel multKernel, int GPU)
        {
            var count = leadingDim * otherDim;

            Debug.Assert(vectors != null && temp != null, "Missing data!");
            Debug.Assert(dotKernel != null && multKernel != null, "Missing kernels.");
            Debug.Assert(leadingDim > 0 && otherDim > 0, "Negative matrix dimensions!");
            Debug.Assert(vectors.Count >= count, "Too little vectors to orthonormalize!");
            Debug.Assert(temp.Count >= Math.Max(leadingDim, otherDim), "Too little temp space!");

            multKernel.SetupExecution(leadingDim);


            for (int i = 0; i < otherDim; i++)
            {
                var seg = vectors.GetDevicePtr(GPU, i * leadingDim);
                //dotKernel.Run(temp, i, seg, seg, leadingDim, /* distributed: */ 0);
                dotKernel.outOffset = i;
                dotKernel.Run(temp, seg, seg, leadingDim);
            }

            temp.SafeCopyToHost(0, otherDim);


            for (int i = 0; i < otherDim; i++)
            {
                if (temp.Host[i] < 0.0000001f)
                    temp.Host[i] = 0;
                else
                    temp.Host[i] = (float)(1 / Math.Sqrt(temp.Host[i]));
            }

            temp.SafeCopyToDevice(0, otherDim);


            for (int i = 0; i < otherDim; i++)
            {
                var seg = vectors.GetDevicePtr(GPU, i * leadingDim);
                var len = temp.GetDevicePtr(GPU, i);
                multKernel.Run(seg, len, seg, (int)MyJoin.MyJoinOperation.Multiplication, leadingDim, 1);
            }
        }

        /// <summary>
        /// Generates a matrix with <paramref name="yDim"/> being the leading dimension in column-major storage.
        /// </summary>
        /// <param name="random">The random object for number generation.</param>
        /// <param name="xDim">The size of the leading dimension.</param>
        /// <param name="yDim">The size of the other dimension.</param>
        /// <param name="orthonormalize">If true, the vectors along the longer dimension will be orthonormalized.</param>
        /// <param name="axisToNormalize">The axis along which to normalize vectors after orthonormalization.</param>
        /// <returns>The generated matrix.</returns>
        public static float[] GenerateTransformMatrix(Random random, int xDim, int yDim, bool orthonormalize = false, AxisToNormalizeEnum axisToNormalize = AxisToNormalizeEnum.yDim)
        {
            Debug.Assert(random != null, "Missing random object");


            var buffer = new float[xDim * yDim];

            // Mapping to rows --- Column-major storage --- rows will the leading dimension
            // The larger dimension vectors will be orthogonal; the cols dimension vectors will be normalized

            if (!orthonormalize)
            {
                // Generate normalized vectors along the cols dim
                if (axisToNormalize == AxisToNormalizeEnum.xDim)
                {
                    GenerateRandomNormalVectors(buffer, random, xDim, yDim);

                    // Transpose to the correct position
                    Transpose(ref buffer, xDim, yDim);
                }
                else
                {
                    GenerateRandomNormalVectors(buffer, random, yDim, xDim);
                }
            }
            else
            {
                int largerDim = Math.Max(xDim, yDim);
                int smallerDim = Math.Min(xDim, yDim);

                // Generate vectors with larger leading dimension
                GenerateRandomNormalVectors(buffer, random, largerDim, smallerDim, normalize: false);
                // Orthonormalize along the larger dimension
                OrthonormalizeVectors(buffer, largerDim, smallerDim);

                if (xDim > yDim)
                {
                    // cols is leading and is normalized
                    // We need to transpose to get the correct dims
                    Transpose(ref buffer, largerDim, smallerDim);

                    if (axisToNormalize == AxisToNormalizeEnum.xDim)
                        NormalizeLeadingDim(buffer, yDim, xDim);
                }
                else
                {
                    // rows is leading and is normalized
                    // The matrix is in correct position

                    if (axisToNormalize == AxisToNormalizeEnum.yDim)
                    {
                        // TODO: SMELLY VERSION:
                        Transpose(ref buffer, yDim, xDim);
                        NormalizeLeadingDim(buffer, xDim, yDim);
                        Transpose(ref buffer, xDim, yDim);
                    }
                }
            }

            return buffer;
        }

        /// <summary>
        /// Generates a matrix with <paramref name="xDim"/> being the leading dimension in column-major storage.
        /// </summary>
        /// <param name="unmanagedVectors">A memory block to store the generated matrix.
        /// Must be as large as <paramref name="xDim"/> x <paramref name="yDim"/>.</param>
        /// <param name="unmanagedBaseVectors">A temporary block to store all the base vectors.
        /// Must be as large as Max(<paramref name="xDim"/>, <paramref name="yDim"/>)^2.
        /// Only neccessary when <paramref name="mode"/> is set to <see cref="VectorGenerationMode.AverageBaseVectors"/>.</param>
        /// <param name="temp">The temporary storage. It should be as long as the longer of the dimensions.</param>
        /// <param name="random">The random object for number generation.</param>
        /// <param name="xDim">The size of the other dimension.</param>
        /// <param name="yDim">The size of the leading dimension.</param>
        /// <param name="mode">If true, the vectors along the longer dimension will be orthonormalized.</param>
        /// <param name="axisToNormalize">The axis along which to normalize vectors after orthonormalization.</param>
        public static void GenerateTransformMatrix(
            MyMemoryBlock<float> unmanagedVectors, MyMemoryBlock<float> unmanagedBaseVectors, MyMemoryBlock<float> temp,
            Random random, int xDim, int yDim,
            MyProductKernel<float> dotKernel, MyCudaKernel multKernel, MyCudaKernel transposeKernel, int GPU,
            VectorGenerationMode mode = VectorGenerationMode.Normal, AxisToNormalizeEnum axisToNormalize = AxisToNormalizeEnum.yDim)
        {
            Debug.Assert(random != null, "Missing random object");
            Debug.Assert(unmanagedVectors != null && (mode != VectorGenerationMode.AverageBaseVectors || unmanagedBaseVectors != null) && temp != null, "Missing data!");
            Debug.Assert(dotKernel != null && multKernel != null && transposeKernel != null, "Missing a kernel!");


            // Mapping to rows --- Column-major storage --- rows will the leading dimension
            // The larger dimension vectors will be orthogonal; the cols dimension vectors will be normalized

            switch (mode)
            {
                case VectorGenerationMode.Normal:
                    if (axisToNormalize == AxisToNormalizeEnum.xDim)
                    {
                        // Generate normalized vectors with xDim as the leading dim
                        GenerateRandomNormalVectors(unmanagedVectors.Host, random, xDim, yDim);
                        unmanagedVectors.SafeCopyToDevice();

                        // Transpose to the correct position
                        transposeKernel.Run(unmanagedVectors, unmanagedVectors, xDim, yDim);
                    }
                    else
                    {
                        GenerateRandomNormalVectors(unmanagedVectors.Host, random, yDim, xDim);
                        unmanagedVectors.SafeCopyToDevice();
                    }
                    break;

                case VectorGenerationMode.Orthonormalize:
                    int largerDim = Math.Max(xDim, yDim);
                    int smallerDim = Math.Min(xDim, yDim);

                    // Generate vectors with larger leading dimension
                    GenerateRandomNormalVectors(unmanagedVectors.Host, random, largerDim, smallerDim, normalize: false);
                    unmanagedVectors.SafeCopyToDevice();

                    // Orthonormalize along the larger dimension
                    OrthonormalizeVectors(unmanagedVectors, temp, largerDim, smallerDim, dotKernel, multKernel, GPU);

                    if (xDim > yDim)
                    {
                        // xDim is leading and is normalized
                        // We need to transpose to get the correct dims
                        transposeKernel.Run(unmanagedVectors, unmanagedVectors, xDim, yDim);

                        if (axisToNormalize == AxisToNormalizeEnum.yDim)
                            NormalizeLeadingDim(unmanagedVectors, temp, yDim, xDim, dotKernel, multKernel, GPU);
                    }
                    else
                    {
                        // yDim is leading and is normalized
                        // The matrix is in correct position

                        if (axisToNormalize == AxisToNormalizeEnum.xDim)
                        {
                            // TODO: generate the matrix with transposed dims?
                            // TODO: SMELLY VERSION:
                            transposeKernel.Run(unmanagedVectors, unmanagedVectors, yDim, xDim);
                            NormalizeLeadingDim(unmanagedVectors, temp, xDim, yDim, dotKernel, multKernel, GPU);
                            transposeKernel.Run(unmanagedVectors, unmanagedVectors, xDim, yDim);
                        }
                    }
                    break;

                case VectorGenerationMode.AverageBaseVectors:
                    int longerDim = Math.Max(xDim, yDim);
                    int shorterDim = Math.Min(xDim, yDim);

                    GenerateTransformMatrix(
                        unmanagedBaseVectors, null, temp,
                        random, longerDim, longerDim,
                        dotKernel, multKernel, transposeKernel, GPU,
                        VectorGenerationMode.Orthonormalize);

                    if (shorterDim == longerDim)
                        break;


                    float it = 0f;
                    float step = longerDim / (float)shorterDim;
                    int beg, end = 0;

                    for (int i = 0; i < shorterDim; i++)
                    {
                        beg = end;
                        it += step;
                        end = (int)it;

                        var vect = unmanagedVectors.GetDevicePtr(GPU, i * longerDim);

                        for (int j = beg; j < end; j++)
                        {
                            var baseVect = unmanagedBaseVectors.GetDevicePtr(GPU, j * longerDim);
                            multKernel.Run(baseVect, vect, vect, (int)MyJoin.MyJoinOperation.Addition, longerDim,
                                longerDim);
                        }
                    }

                    if (xDim > yDim)
                    {
                        // xDim is leading and is not normalized
                        // We need to transpose to get the correct dims

                        if (axisToNormalize == AxisToNormalizeEnum.xDim)
                        {
                            NormalizeLeadingDim(unmanagedVectors, temp, xDim, yDim, dotKernel, multKernel, GPU);

                            transposeKernel.Run(unmanagedVectors, unmanagedVectors, xDim, yDim);
                        }
                        else
                        {
                            transposeKernel.Run(unmanagedVectors, unmanagedVectors, xDim, yDim);

                            NormalizeLeadingDim(unmanagedVectors, temp, yDim, xDim, dotKernel, multKernel, GPU);
                        }
                    }
                    else
                    {
                        // yDim is leading and is not normalized
                        // The matrix is in correct position

                        if (axisToNormalize == AxisToNormalizeEnum.yDim)
                            NormalizeLeadingDim(unmanagedVectors, temp, yDim, xDim, dotKernel, multKernel, GPU);
                        else
                        {
                            // TODO: SMELLY VERSION:
                            transposeKernel.Run(unmanagedVectors, unmanagedVectors, yDim, xDim);
                            NormalizeLeadingDim(unmanagedVectors, temp, xDim, yDim, dotKernel, multKernel, GPU);
                            transposeKernel.Run(unmanagedVectors, unmanagedVectors, xDim, yDim);
                        }
                    }
                    break;
            }
        }


        /// <summary>
        /// Computes the inner product of two vectors specified by the array segments.
        /// </summary>
        /// <param name="first">The first vector of the inner product.</param>
        /// <param name="second">The second vector of the inner product.</param>
        /// <returns>The inner products of the given vectors.</returns>
        public static double DotProduct(ArraySegment<float> first, ArraySegment<float> second)
        {
            Debug.Assert(first.Count == second.Count, "Invalid vector dimenstions!");


            double sum = 0;

            for (int i = 0; i < first.Count; i++)
                sum += first.Array[first.Offset + i] * second.Array[second.Offset + i];

            return sum;
        }

        /// <summary>
        /// Transforms all the vectors stored in <paramref name="vectors"/> to be pair-wise orthonormal using a modified version of the Gram-Schmidt algorithm.
        /// </summary>
        /// <param name="vectors">The vectors to orthonormalize.</param>
        /// <param name="xDim">The length of each vector.</param>
        /// <param name="yDim">The number of vectors.</param>
        public static void OrthonormalizeVectors(float[] vectors, int xDim, int yDim)
        {
            Debug.Assert(xDim > 0 && yDim > 0, "Negative matrix dimensions!");
            Debug.Assert(xDim * yDim <= vectors.Length, "Invalid dimensions!");


            for (int i = 0; i < xDim * yDim; i += xDim)
            {
                var currSeg = new ArraySegment<float>(vectors, i, xDim);

                // Normalize the current vector
                {
                    double dot = DotProduct(currSeg, currSeg);

                    if (dot < 0.0000001f)
                        continue;

                    float fdot = (float)(1 / Math.Sqrt(dot));

                    for (int j = i; j < i + xDim; j++)
                        vectors[j] *= fdot;
                }


                // Make all the remaining vectors orthogonal to the current one
                for (int j = i + xDim; j < vectors.Length; j += xDim)
                {
                    var nextSeg = new ArraySegment<float>(vectors, j, xDim);

                    // Compute and subtract the projection onto the current vector
                    float dot = (float)DotProduct(currSeg, nextSeg);

                    for (int k = 0; k < xDim; k++)
                        vectors[j + k] -= dot * vectors[i + k];
                }
            }
        }

        /// <summary>
        /// Transforms all the vectors stored in <paramref name="vectors"/> to be pair-wise orthonormal using a modified version of the Gram-Schmidt algorithm.
        /// </summary>
        /// <param name="vectors">The vectors to orthonormalize.</param>
        /// <param name="temp">A vector of temporal space.</param>
        /// <param name="xDim">The length of each vector.</param>
        /// <param name="yDim">The number of vectors.</param>
        /// <param name="dotKernel">The kernel to compute a dot product.</param>
        /// <param name="multKernel">The kernel to compute vector combinations.</param>
        public static void OrthonormalizeVectors(MyMemoryBlock<float> vectors, MyMemoryBlock<float> temp, int xDim, int yDim, MyProductKernel<float> dotKernel, MyCudaKernel multKernel, int GPU)
        {
            int count = xDim * yDim;

            Debug.Assert(vectors != null && temp != null, "Missing data!");
            Debug.Assert(dotKernel != null && multKernel != null, "Missing a kernel!");
            Debug.Assert(xDim > 0 && yDim > 0, "Negative matrix dimensions!");
            Debug.Assert(vectors.Count >= count, "Too little vectors to orthonormalize!");
            Debug.Assert(temp.Count >= xDim, "Too little temp space!");

            multKernel.SetupExecution(xDim);


            for (int i = 0; i < count; i += xDim)
            {
                var curr = vectors.GetDevicePtr(GPU, i);

                // Normalize the current vector
                {
                    //ZXC dotKernel.Run(temp, 0, curr, curr, xDim, /* distributed: */ 0);
                    dotKernel.Run(temp, curr, curr, xDim);
                    temp.SafeCopyToDevice(0, 1);

                    if (temp.Host[0] < 0.0000001f)
                        continue;

                    temp.Host[0] = (float)(1 / Math.Sqrt(temp.Host[0]));
                    temp.SafeCopyToDevice(0, 1);

                    multKernel.Run(curr, temp, curr, (int)MyJoin.MyJoinOperation.Multiplication, xDim, 1);
                }

                // Make all the remaining vectors orthogonal to the current one
                for (int j = i + xDim; j < count; j += xDim)
                {
                    var next = vectors.GetDevicePtr(GPU, j);

                    // Compute and subtract the projection onto the current vector
                    //ZXC dotKernel.Run(temp, xDim, curr, next, xDim, /* distributed: */ 0);
                    dotKernel.outOffset = xDim;
                    dotKernel.Run(temp, curr, next, xDim);

                    multKernel.Run(curr, temp, temp, (int)MyJoin.MyJoinOperation.Multiplication, xDim, 1);
                    multKernel.Run(next, temp, next, (int)MyJoin.MyJoinOperation.Subtraction, xDim, xDim);
                }
            }
        }

        #endregion
    }
}
