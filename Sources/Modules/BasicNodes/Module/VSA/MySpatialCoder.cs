using BrainSimulator.Memory;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace BrainSimulator.VSA
{
    /// <author>Dusan Fedorcak</author>
    /// <status>WIP</status>
    /// <summary>Encodes &amp; decodes spatial values into symbols through linear interpolation</summary>
    /// <description></description>
    public class MySpatialCoder : MyCodeBookBase
    {
        [MyOutputBlock(1)]
        public MyMemoryBlock<float> Reliability
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        public enum MySpatialCoderMode
        {
            Encode,
            Decode,            
        }

        [MyBrowsable, Category("Grid")]
        [YAXSerializableField(DefaultValue = MySpatialCoderMode.Encode)]
        public MySpatialCoderMode Mode { get; set; }

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

            if (Input != null)
            {
                switch (Mode)
                {
                    case MySpatialCoderMode.Encode:
                        {
                            validator.AssertError(Input.Count > 0, this, "Input must be greater then 0.");
                        }
                        break;
                    case MySpatialCoderMode.Decode:                    
                        {
                            validator.AssertError(Input.Count == SymbolSize, this, "Input must be equal to symbol size.");
                        }
                        break;
                }
            }
        }

        public MySpatialCoderTask DoTransform { get; private set; }

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
                CudaDeviceVariable<float> codeVectors = MyMemoryManager.Instance.GetGlobalVariable<float>(
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
