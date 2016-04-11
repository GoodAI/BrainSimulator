using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.VSA;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Testing
{
    /// <author>GoodAI</author>
    /// <meta>mv,mm</meta>
    /// <status>temporary</status>
    /// <summary>Converts data from KMeansWM to Symbol rep.</summary>
    /// <description>
    /// This is a temporary project-specific node hardly usable for more general purposes and will be removed with a future update.
    /// Used in the Breakout-playing AI project.
    /// </description>
    public class MyKWM2SymbolNode : MyCodeBookBase
    {
        [MyInputBlock(1)]
        public MyMemoryBlock<float> ClustersPos
        {
            get { return GetInput(1); }
        }

        public MyMemoryBlock<float> Temp { get; private set; }
        public MyMemoryBlock<float> SpatialTemp { get; private set; }
        public MyMemoryBlock<float> SpatialCode { get; private set; }
        public MyMemoryBlock<float> PositionSymbol { get; private set; }
        public MyMemoryBlock<float> InterResult { get; private set; }
        public MyMemoryBlock<float> ActualCluster { get; private set; }

        public MyKMeansWMInitTask Init { get; private set; }

        public int ClusterSize
        {
            get { return Input != null ? Input.ColumnHint : 0; }
        }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 4)]
        public int MaxClusters { get; set; }

        public override void UpdateMemoryBlocks()
        {
            SpatialCode.Count = ClusterSize;
            Temp.Count = MyFourierBinder.GetTempBlockSize(ClusterSize);
            Output.Count = ClusterSize;
            PositionSymbol.Count = ClusterSize;
            ActualCluster.Count = ClusterSize;
            InterResult.Count = ClusterSize;
            SpatialTemp.Count = ClusterSize;

            SpatialCode.ColumnHint = Temp.ColumnHint = Output.ColumnHint = SpatialTemp.ColumnHint = ColumnHint;
            PositionSymbol.ColumnHint = ActualCluster.ColumnHint = InterResult.ColumnHint = ColumnHint;
        }

        [MyTaskInfo(OneShot = false), Description("Init")]
        public class MyKMeansWMInitTask : MyTask<MyKWM2SymbolNode>
        {
            private MySymbolBinderBase m_binder;
            private MyCudaKernel m_spatialKernel;
            private MyCudaKernel m_addKernel;

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = false)]
            public bool UseSquaredTransform { get; set; }

            public override void Init(int nGPU)
            {
                if (Owner.UseBSCVariety)
                    m_binder = new MyXORBinder(Owner, Owner.ClusterSize);
                else
                    m_binder = new MyFourierBinder(Owner, Owner.ClusterSize, Owner.Temp);

                m_spatialKernel = MyKernelFactory.Instance.Kernel(nGPU, @"VSA\SpatialCoder", "EncodeValues");
                m_spatialKernel.SetupExecution(Owner.ClusterSize);

                m_addKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineBinaryVectorsKernel",
                    "CombineTwoVectorsKernel");
                m_addKernel.SetupExecution(Owner.ClusterSize);
            }

            public override void Execute()
            {
                if (SimulationStep < 1)
                {
                    GetPositionSymbol();
                }

                Owner.Output.Fill(0f);

                int nrClusters = Math.Min(Owner.Input.Count / Owner.ClusterSize, Owner.MaxClusters);
                Owner.ClustersPos.SafeCopyToHost();

                var interResult = Owner.InterResult.GetDevicePtr(Owner.GPU);
                var positionSymbol = Owner.PositionSymbol.GetDevicePtr(Owner.GPU);

                for (int i = 0; i < nrClusters; ++i)
                {
                    float x1 = Owner.ClustersPos.Host[i * Owner.ClustersPos.ColumnHint];
                    float y1 = Owner.ClustersPos.Host[i * Owner.ClustersPos.ColumnHint + 1];
                    //MyLog.DEBUG.WriteLine(x + " - " + y);

                    var space1 = SpatialEncode(x1, y1);
                    var shape1 = Owner.Input.GetDevicePtr(Owner.GPU, i * Owner.ClusterSize);

                    m_binder.Bind(shape1, positionSymbol, interResult);
                    m_binder.Bind(space1, interResult, interResult);

                    m_addKernel.Run(Owner.Output, Owner.InterResult, Owner.Output, 0, Owner.Output.Count);

                    // Compute relative differences
                    for (int j = i + 1; j < nrClusters; j++)
                    {
                        float x2 = Owner.ClustersPos.Host[j * Owner.ClustersPos.ColumnHint];
                        float y2 = Owner.ClustersPos.Host[j * Owner.ClustersPos.ColumnHint + 1];

                        var shape2 = Owner.Input.GetDevicePtr(Owner.GPU, j * Owner.ClusterSize);
                        var space2 = SpatialEncode(x1 - x2, y1 - y2);

                        m_binder.Unbind(shape1, shape2, interResult);
                        m_binder.Bind(positionSymbol, interResult, interResult);
                        m_binder.Bind(space2, interResult, interResult);

                        m_addKernel.Run(Owner.Output, Owner.InterResult, Owner.Output, 0, Owner.Output.Count);

                        space2 = SpatialEncode(x2 - x1, y2 - y1);

                        m_binder.Unbind(shape2, shape1, interResult);
                        m_binder.Bind(positionSymbol, interResult, interResult);
                        m_binder.Bind(space2, interResult, interResult);

                        m_addKernel.Run(Owner.Output, Owner.InterResult, Owner.Output, 0, Owner.Output.Count);
                    }
                }
            }

            private int GetSymbolOffset(MyCodeVector symbol)
            {
                return (int)symbol * Owner.ClusterSize * sizeof(float);
            }

            private void GetPositionSymbol()
            {
                CudaDeviceVariable<float> codeVector = new CudaDeviceVariable<float>(
                    MyMemoryManager.Instance.GetGlobalVariable<float>(Owner.GlobalVariableName, Owner.GPU, Owner.GenerateRandomVectors).DevicePointer
                    + (int)MyCodeVector.Position * Owner.SymbolSize * sizeof(float), Owner.SymbolSize);

                Owner.PositionSymbol.GetDevice(Owner).CopyToDevice(codeVector, 0, 0, sizeof(float) * Owner.ClusterSize);
            }

            private CUdeviceptr SpatialEncode(float x, float y)
            {
                Owner.SpatialTemp.Host[0] = x;
                Owner.SpatialTemp.Host[1] = y;
                Owner.SpatialTemp.SafeCopyToDevice();

                CudaDeviceVariable<float> codeVectors = MyMemoryManager.Instance.GetGlobalVariable<float>(
                    Owner.GlobalVariableName, Owner.GPU, Owner.GenerateRandomVectors);

                CUdeviceptr dirX = codeVectors.DevicePointer + GetSymbolOffset(MyCodeVector.DirX);
                CUdeviceptr dirY = codeVectors.DevicePointer + GetSymbolOffset(MyCodeVector.DirY);
                CUdeviceptr negDirX = codeVectors.DevicePointer + GetSymbolOffset(MyCodeVector.NegDirX);
                CUdeviceptr negDirY = codeVectors.DevicePointer + GetSymbolOffset(MyCodeVector.NegDirY);

                CUdeviceptr originX = codeVectors.DevicePointer + GetSymbolOffset(MyCodeVector.OriginX);
                CUdeviceptr originY = codeVectors.DevicePointer + GetSymbolOffset(MyCodeVector.OriginY);

                m_spatialKernel.Run(Owner.SpatialTemp, 2, Owner.SpatialCode, Owner.ClusterSize, UseSquaredTransform ? 1 : 0,
                        dirX, dirY, negDirX, negDirY, originX, originY);

                return Owner.SpatialCode.GetDevicePtr(Owner.GPU);
            }
        }
    }
}
