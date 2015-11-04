using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.VSA
{
    /// <author>GoodAI</author>
    /// <meta>df</meta>
    /// <status>working</status>
    /// <summary>Encodes &amp; decodes spatial values into symbols through linear interpolation.</summary>
    /// <description>
    /// 
    /// <h3> Features: </h3>
    /// Transforms the input based on one of the following modes:
    /// <ol>
    ///     <li><h4>Encode:</h4></li>
    ///     <ul>
    ///         <li>Expects (or rather works best with) a vector of floats in the real interval <code>[-1, 1]</code>. Note that a maximum of 2 numbers can be encoded with this node.</li>
    ///         <li>For the value <code>t</code> in each dimension, computes the encoding as</li>
    ///             <code>t*dir + (1-t)*o</code>,<br/>
    ///             where <code>dir</code> is either <code>negDir</code> if <code>t&lt;0</code> or <code>posDir</code> otherwise.
    ///             Vectors <code>negDir</code>, <code>posDir</code> and <code>o</code> are random precomputed symbolic
    ///             constants specific for the dimension (globally shared -- you can access them via the CodeBook node).
    ///         <li>Superposes the encodings for each dimension and outputs it.</li>
    ///     </ul>
    /// 
    ///     <li><h4>Decode:</h4></li>
    ///     <ul>
    ///         <li>Expects a vector s with superposed values encoded by this node.</li>
    ///         <li>For each dimension of the output, computes</li>
    ///             <code>s.posDir - s.negDir = s.dir</code> (the symbol . denoting the dot product).<br/>
    ///             Note that one of the values <code>s.posDir</code> or <code>s.negDir</code> will be zero.
    ///         <li>Computes the reliability of the decoding as</li>
    ///             <code>rel = s.o + s.dir</code><br/>
    ///             This value indicates the amount of noise in the input vector (since for orthogonal
    ///             <code>o</code> and <code>dir</code> and no noise in the superposition, <code>rel</code> should sum up to 1).
    ///         <li>Outputs the value</li>
    ///             <code>output = s.dir / rel</code><br/>
    ///             as the result of decoding for the corresponding dimension.
    ///         <br/>
    ///         <h4>Sidenotes:</h4>
    ///         <li>Let's say that</li> 
    ///             <code>s = dir*t + o*(1-t) + noise</code>
    ///         <li>We then get:</li>
    ///             <code>s.dir &#160;&#160;= dir.dir*t + o.dir*(1-t) + dir.noise = t + 0 + dir.noise</code><br/>
    ///             <code>s.o &#160;&#160;&#160;&#160;= o.dir*t &#160;&#160;+ o.o*(1-t) &#160;&#160;+ o.noise &#160;&#160;= 0 + (1-t) + o.noise</code><br/>
    ///             <code>output &#160;= t + dir.noise / (1 + dir.noise + o.noise)</code>
    ///         <li>Now <code>dir.noise</code> and <code>o.noise</code> should be very close to zero. This decoding approach should make
    ///             the decoding more precise when <code>noise</code> is has a similar dot product to <code>dir</code> and <code>o</code>.</li>
    ///     </ul>
    /// </ol>
    /// 
    /// <h3> Important notices: </h3>
    /// <ul>
    ///     <li>This node splits the interval <code>[-1, 1]</code> into two parts and interpolates in each of them.
    ///         To split the interval into arbitrarily many parts, use the SpatialGrid node.
    ///         The benefit is that the hyperspace gets more evenly occupied by interpolating between many more than 3 points,
    ///         but the computation is much more intensive, because we are working with many more vectors.</li>
    /// </ul>
    /// </description>
    public class MySpatialCoder : MyCodeBookBase
    {
        public enum MySpatialCoderMode
        {
            Encode,
            Decode,
        }


        [MyOutputBlock(1)]
        public MyMemoryBlock<float> Reliability
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }


        [MyBrowsable, Category("Grid"), Description("Specifies whether to encode input scalars or decode input vector.")]
        [YAXSerializableField(DefaultValue = MySpatialCoderMode.Encode)]
        public MySpatialCoderMode Mode { get; set; }


        #region MyNode overrides

        public override string Description
        {
            get
            {
                switch (Mode)
                {
                    case MySpatialCoderMode.Encode: return "(x,y)->symbol";
                    case MySpatialCoderMode.Decode: return "symbol->(x,y)";
                    default: return "N/A";
                }
            }
        }

        public override void UpdateMemoryBlocks()
        {
            Reliability.Count = 2;

            if (Mode == MySpatialCoderMode.Decode)
            {
                Output.Count = 2;
                Output.ColumnHint = 1;
            }
            else
            {
                Output.Count = SymbolSize;
                Output.ColumnHint = ColumnHint;
            }
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (Input == null)
                return;

            switch (Mode)
            {
                case MySpatialCoderMode.Encode:
                    {
                        validator.AssertError(Input.Count > 0, this, "In encoding mode, the input size must be greater than 0.");
                    }
                    break;
                case MySpatialCoderMode.Decode:
                    {
                        validator.AssertError(Input.Count == SymbolSize, this, "In decoding mode, the input count must be equal to symbol size.");
                    }
                    break;
            }
        }

        #endregion


        public MySpatialCoderTask DoTransform { get; private set; }

        /// <summary>
        /// Performs the encoding or decoding of the input.
        /// </summary>
        public class MySpatialCoderTask : MyTask<MySpatialCoder>
        {
            private MyCudaKernel m_kernel;

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = false)]
            public bool UseSquaredTransform { get; set; }

            public override void Init(int nGPU)
            {
                if (Owner.Mode == MySpatialCoderMode.Encode)
                {
                    m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"VSA\SpatialCoder", "EncodeValues");
                    m_kernel.SetupExecution(Owner.SymbolSize);
                }
                else
                {
                    m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"VSA\SpatialCoder", "DecodeValues");
                    m_kernel.SetupExecution(Owner.Output.Count);
                }
            }

            public override void Execute()
            {
                CudaDeviceVariable<float> codeVectors = MyMemoryManager.Instance.GetGlobalVariable(
                    Owner.GlobalVariableName, Owner.GPU, Owner.GenerateRandomVectors);

                CUdeviceptr dirX = codeVectors.DevicePointer + GetSymbolOffset(MyCodeVector.DirX);
                CUdeviceptr dirY = codeVectors.DevicePointer + GetSymbolOffset(MyCodeVector.DirY);
                CUdeviceptr negDirX = codeVectors.DevicePointer + GetSymbolOffset(MyCodeVector.NegDirX);
                CUdeviceptr negDirY = codeVectors.DevicePointer + GetSymbolOffset(MyCodeVector.NegDirY);

                CUdeviceptr originX = codeVectors.DevicePointer + GetSymbolOffset(MyCodeVector.OriginX);
                CUdeviceptr originY = codeVectors.DevicePointer + GetSymbolOffset(MyCodeVector.OriginY);

                if (Owner.Mode == MySpatialCoderMode.Encode)
                {
                    m_kernel.Run(Owner.Input, Owner.Input.Count, Owner.Output, Owner.SymbolSize, UseSquaredTransform ? 1 : 0,
                        dirX, dirY, negDirX, negDirY, originX, originY);
                }
                else
                {
                    m_kernel.Run(Owner.Input, Owner.SymbolSize, Owner.Output, Owner.Reliability, Owner.Output.Count, UseSquaredTransform ? 1 : 0,
                        dirX, dirY, negDirX, negDirY, originX, originY);
                }
            }

            private int GetSymbolOffset(MyCodeVector symbol)
            {
                return (int)symbol * Owner.SymbolSize * sizeof(float);
            }
        }
    }
}
