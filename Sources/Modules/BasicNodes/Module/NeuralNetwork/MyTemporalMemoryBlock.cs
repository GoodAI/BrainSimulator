using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using GoodAI.Core.Nodes;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;

namespace GoodAI.Modules.NeuralNetwork
{
    public class MyTemporalMemoryBlock<T> : MyMemoryBlock<T> where T : struct
    {
        public override MyNode Owner
        {
            get { return m_owner; }
            set
            {
                if ((value != null) && !(value is MyAbstractLayer))
                {
                    throw new InvalidOperationException(
                        "Temporal memory blocks can be used only inside nodes inherited from MyAbstractLayer");
                }

                m_owner = value;
            }
        }
        private MyNode m_owner;

        private MyNeuralNetworkGroup ParentNetwork
        {
            get
            {
                if (Owner == null)
                    throw new NullReferenceException("Owner not set.");

                var layer = Owner as MyAbstractLayer;
                if (layer == null)
                    throw new InvalidOperationException("Owner is not a MyAbstractLayer");

                MyNeuralNetworkGroup parentNetwork = layer.ParentNetwork;
                if (parentNetwork == null)
                {
                    throw new NullReferenceException("Owner.ParentNetwork not set.");
                }

                return parentNetwork;
            }
        }

        public int SequenceLength
        {
            get
            {
                return ParentNetwork.SequenceLength;
            }
        }
        
        public int BoundedSequenceLength
        {
            get
            {
                // +2 because there are two empty boundary blocks one before, one after
                // however they are physically both at the end of memory
                return SequenceLength + 2;
            }
        }

        public int TimeStep
        {
            get
            {
                int timeStep = ParentNetwork.TimeStep;
                if (0 <= timeStep && timeStep < SequenceLength)
                    return timeStep;

                return SequenceLength - 1;
            }
        }

        public int TimeOffset
        {
            get
            {
                return TimeStep * Count;
            }
        }

        private MyCudaKernel modeKernel;

        public enum ModeType
        {
            None,
            Cumulate,
            Copy
        }

        public ModeType Mode
        {
            get
            {
                return m_mode;
            }
            set
            {
                m_mode = value;
                switch (m_mode)
                {
                    case ModeType.None:
                        modeKernel = null;
                        return;
                    case ModeType.Cumulate:
                        modeKernel = MyKernelFactory.Instance.Kernel(Owner.GPU, @"NeuralNetwork\TemporalMemoryBlock", "CumulateThroughTimeKernel");
                        break;
                    case ModeType.Copy:
                        modeKernel = MyKernelFactory.Instance.Kernel(Owner.GPU, @"NeuralNetwork\TemporalMemoryBlock", "CopyThroughTimeKernel");
                        break;
                }
                modeKernel.SetupExecution(Count);
            }
        }
        private ModeType m_mode;

        public void RunMode()
        {
            if (modeKernel != null)
                modeKernel.Run(GetDevicePtr(Owner.GPU, 0, 0), Count, SequenceLength);
        }

        public CUdeviceptr GetTimeShiftedBlock(int timeShift)
        {
            int t = TimeStep + timeShift;
            if (t < 0) // get boundary block at beginning
                return GetDevicePtr(Owner.GPU, 0, BoundedSequenceLength - 2);
            else if (SequenceLength <= t) // get boundary block at the end
                return GetDevicePtr(Owner.GPU, 0, BoundedSequenceLength - 1);
            return GetDevicePtr(Owner.GPU, 0, t);
        }

        public override void AllocateDevice()
        {
            if (Count > 0)
            {
                if (Device == null)
                {
                    Device = new CudaDeviceVariable<T>[MyKernelFactory.Instance.DevCount];

                    if (!Unmanaged)
                    {
                        MyLog.DEBUG.WriteLine("Allocating: " + typeof(T).ToString() + ", " + Count * BoundedSequenceLength * System.Runtime.InteropServices.Marshal.SizeOf(typeof(T)));
                        Device[Owner.GPU] = new CudaDeviceVariable<T>(
                           MyKernelFactory.Instance.GetContextByGPU(Owner.GPU).AllocateMemory(
                           Count * BoundedSequenceLength * System.Runtime.InteropServices.Marshal.SizeOf(typeof(T))));

                        Fill(0);
                    }
                    else
                    {
                        if (ExternalPointer != 0)
                        {
                            Device[Owner.GPU] = new CudaDeviceVariable<T>(new CUdeviceptr(ExternalPointer), Count * BoundedSequenceLength * sizeof(float));
                        }
                        else
                        {
                            throw new ArgumentOutOfRangeException("External Pointer not set for Unmanaged memory block.");
                        }
                    }
                }
            }
        }

        public override bool SafeCopyToDevice()
        {
            if (!OnDevice)
            {
                AllocateDevice();
            }

            if (OnDevice)
            {
                int size = Marshal.SizeOf(typeof(T));
                Device[Owner.GPU].CopyToDevice(Host, 0, size * TimeOffset, size * Count);
                return true;
            }
            else return false;
        }

        public override bool SafeCopyToDevice(int offset, int length)
        {
            if (!OnDevice)
            {
                AllocateDevice();
            }

            if (OnDevice)
            {
                int size = Marshal.SizeOf(typeof(T));
                Device[Owner.GPU].CopyToDevice(Host, size * offset, size * (TimeOffset + offset), size * length);
                return true;
            }
            else return false;
        }

        public override void SafeCopyToHost()
        {
            if (!OnHost)
            {
                AllocateHost();
            }

            if (OnDevice)
            {
                int size = Marshal.SizeOf(typeof(T));
                Device[Owner.GPU].CopyToHost(Host, size * TimeOffset, 0, size * Count);
            }
        }

        public override void SafeCopyToHost(int offset, int length)
        {
            if (!OnHost)
            {
                AllocateHost();
            }

            if (OnDevice)
            {
                int size = Marshal.SizeOf(typeof(T));
                Device[Owner.GPU].CopyToHost(Host, size * (TimeOffset + offset), size * offset, size * length);
            }
        }

        public override void CopyFromMemoryBlock(MyMemoryBlock<T> source, int srcOffset, int destOffset, int count)
        {
            int size = Marshal.SizeOf(typeof(T));
            if (source is MyTemporalMemoryBlock<T>)
                Device[Owner.GPU].CopyToDevice(source.GetDevice(Owner.GPU), (TimeOffset + srcOffset) * size, (TimeOffset + destOffset) * size, count * size);
            else
                Device[Owner.GPU].CopyToDevice(source.GetDevice(Owner.GPU), srcOffset * size, (TimeOffset + destOffset) * size, count * size);
        }

        public override void CopyToMemoryBlock(MyMemoryBlock<T> destination, int srcOffset, int destOffset, int count)
        {
            int size = Marshal.SizeOf(typeof(T));
            if (destination is MyTemporalMemoryBlock<T>)
                destination.GetDevice(Owner.GPU).CopyToDevice(Device[Owner.GPU], (TimeOffset + srcOffset) * size, (TimeOffset + destOffset) * size, count * size);
            else
                destination.GetDevice(Owner.GPU).CopyToDevice(Device[Owner.GPU], (TimeOffset + srcOffset) * size, destOffset * size, count * size);
        }

        public override CudaDeviceVariable<T> GetDevice(MyWorkingNode callee)
        {
            return GetDevice(callee.GPU);
        }

        public override CudaDeviceVariable<T> GetDevice(int nGPU)
        {
            if (OnDevice)
            {
                if (nGPU == Owner.GPU)
                {
                    return Device[Owner.GPU];
                }
                else
                {
                    if (Device[nGPU] == null)
                    {
                        Device[nGPU] = new CudaDeviceVariable<T>(
                            MyKernelFactory.Instance.GetContextByGPU(nGPU).AllocateMemory(
                            BoundedSequenceLength * Count * Marshal.SizeOf(typeof(T))));

                        CopyToGPU(nGPU);
                        Shared = true;
                    }
                    return Device[nGPU];
                }
            }
            else
                return null;
        }

        public override CUdeviceptr GetDevicePtr(int GPU)
        {
            return GetDevicePtr(GPU, 0);
        }

        public override CUdeviceptr GetDevicePtr(int GPU, int offset)
        {
            return GetDevicePtr(GPU, offset, -1);
        }

        public override CUdeviceptr GetDevicePtr(int GPU, int offset, int memBlockIdx)
        {
            CudaDeviceVariable<T> rDeviceVar = GetDevice(GPU);
            if (0 <= memBlockIdx && memBlockIdx < BoundedSequenceLength)
                return rDeviceVar != null ? rDeviceVar.DevicePointer + (memBlockIdx * Count) * rDeviceVar.TypeSize : default(CUdeviceptr);
            return rDeviceVar != null ? rDeviceVar.DevicePointer + (TimeOffset + offset) * rDeviceVar.TypeSize : default(CUdeviceptr);
        }

        public override CUdeviceptr GetDevicePtr(MyAbstractObserver callee, int offset)
        {
            CudaDeviceVariable<T> rDeviceVar = GetDevice(MyKernelFactory.Instance.DevCount - 1);
            return rDeviceVar != null ? rDeviceVar.DevicePointer + (TimeOffset + offset) * rDeviceVar.TypeSize : default(CUdeviceptr);
        }

        public void FillAll(float value)
        {
            CudaDeviceVariable<T> rDeviceVar = GetDevice(Owner.GPU);
            CudaDeviceVariable<T> rTimeOffsettedDeviceVar = new CudaDeviceVariable<T>(rDeviceVar.DevicePointer, false, BoundedSequenceLength * GetSize());
            rTimeOffsettedDeviceVar.Memset(BitConverter.ToUInt32(BitConverter.GetBytes(value), 0));
        }

        public override void Fill(float value)
        {
            CudaDeviceVariable<T> rDeviceVar = GetDevice(Owner.GPU);
            CUdeviceptr timeOffsettedPtr = rDeviceVar.DevicePointer + TimeOffset * rDeviceVar.TypeSize;
            CudaDeviceVariable<T> rTimeOffsettedDeviceVar = new CudaDeviceVariable<T>(timeOffsettedPtr, false, GetSize());
            rTimeOffsettedDeviceVar.Memset(BitConverter.ToUInt32(BitConverter.GetBytes(value), 0));
        }

        public override void Fill(int value)
        {
            CudaDeviceVariable<T> rDeviceVar = GetDevice(Owner.GPU);
            CUdeviceptr timeOffsettedPtr = rDeviceVar.DevicePointer + TimeOffset * rDeviceVar.TypeSize;
            CudaDeviceVariable<T> rTimeOffsettedDeviceVar = new CudaDeviceVariable<T>(timeOffsettedPtr, false, GetSize());
            rTimeOffsettedDeviceVar.Memset(BitConverter.ToUInt32(BitConverter.GetBytes(value), 0));
        }

        public override void Fill(uint value)
        {
            CudaDeviceVariable<T> rDeviceVar = GetDevice(Owner.GPU);
            CUdeviceptr timeOffsettedPtr = rDeviceVar.DevicePointer + TimeOffset * rDeviceVar.TypeSize;
            CudaDeviceVariable<T> rTimeOffsettedDeviceVar = new CudaDeviceVariable<T>(timeOffsettedPtr, false, GetSize());
            rTimeOffsettedDeviceVar.Memset(BitConverter.ToUInt32(BitConverter.GetBytes(value), 0));
        }

        public override void Fill(bool value)
        {
            CudaDeviceVariable<T> rDeviceVar = GetDevice(Owner.GPU);
            CUdeviceptr timeOffsettedPtr = rDeviceVar.DevicePointer + TimeOffset * rDeviceVar.TypeSize;
            CudaDeviceVariable<T> rTimeOffsettedDeviceVar = new CudaDeviceVariable<T>(timeOffsettedPtr, false, GetSize());
            rTimeOffsettedDeviceVar.Memset(BitConverter.ToUInt32(BitConverter.GetBytes(value), 0));
        }

        public override T GetValueAt(int index)
        {
            T value = new T();

            if (OnDevice)
            {
                Device[Owner.GPU].CopyToHost(ref value, (TimeOffset + index) * Marshal.SizeOf(typeof(T)));
            }

            return value;
        }
    }
}
