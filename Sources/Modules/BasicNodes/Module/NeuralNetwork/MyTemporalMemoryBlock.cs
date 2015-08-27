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
        public int SequenceLength
        {
            get
            {
                if (Owner is MyAbstractLayer)
                {
                    MyAbstractLayer layer = Owner as MyAbstractLayer;
                    // +1 because there is one empty block at time [SequenceLength]
                    return layer.ParentNetwork.SequenceLength + 1;
                }
                throw new Exception("TimeMemoryBlocks can be used only inside nodes inherited from MyAbstractLayer");
            }
        }

        public int TimeStep
        {
            get
            {
                if (Owner is MyAbstractLayer)
                {
                    MyAbstractLayer layer = Owner as MyAbstractLayer;
                    return layer.ParentNetwork.TimeStep;
                }
                throw new Exception("TimeMemoryBlocks can be used only inside nodes inherited from MyAbstractLayer");
            }
        }

        public int TimeOffset
        {
            get
            {
                return TimeStep * Count;
            }
        }

        public enum ModeType
        {
            None,
            Cumulate
        }
        public ModeType Mode = ModeType.None;

        public void RunMode()
        {
            switch (Mode)
            {
                case ModeType.None: break;
                case ModeType.Cumulate:
                {
                    CumulateThroughTime();
                    break;
                }
                default: break;
            }
        }

        private void CumulateThroughTime()
        {
            if (typeof(T) == typeof(float))
            {
                int size = Marshal.SizeOf(typeof(T));

                // make it efficient
                T[] HostAtTimeZero = new T[Count];
                Device[Owner.GPU].CopyToHost(HostAtTimeZero, 0, 0, size * Count);
                for (int t = 1; t < SequenceLength-1; t++)
			    {
                    Device[Owner.GPU].CopyToHost(Host, size * t * Count, 0, size * Count);
                    for (int i = 0; i < HostAtTimeZero.Length; i++)
			        {
                        // C# thingy...
                        float x = (float)(object)HostAtTimeZero[i];
                        x += (float)(object)Host[i];
                        HostAtTimeZero[i] = (T)(object)x;
			        }
			    }
                Device[Owner.GPU].CopyToDevice(HostAtTimeZero, 0, 0, size * Count);
            }
        }

        public CUdeviceptr GetTimeShiftedBlock(int timeShift)
        {
            int t = TimeStep + timeShift;
            if (t <= -1 || t > SequenceLength)
                return GetDevicePtr(Owner.GPU, 0, SequenceLength-1);
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
                        MyLog.DEBUG.WriteLine("Allocating: " + typeof(T).ToString() + ", " + Count * SequenceLength * System.Runtime.InteropServices.Marshal.SizeOf(typeof(T)));
                        Device[Owner.GPU] = new CudaDeviceVariable<T>(
                           MyKernelFactory.Instance.GetContextByGPU(Owner.GPU).AllocateMemory(
                           Count * SequenceLength * System.Runtime.InteropServices.Marshal.SizeOf(typeof(T))));

                        Fill(0);
                    }
                    else
                    {
                        if (ExternalPointer != 0)
                        {
                            Device[Owner.GPU] = new CudaDeviceVariable<T>(new CUdeviceptr(ExternalPointer), Count * SequenceLength * sizeof(float));
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
                            SequenceLength * Count * Marshal.SizeOf(typeof(T))));

                        CopyToGPU(nGPU);
                        Shared = true;
                    }
                    return Device[nGPU];
                }
            }
            else
                return null;
        }

        public override CUdeviceptr GetDevicePtr(int GPU, int offset, int timeStep = -1)
        {
            CudaDeviceVariable<T> rDeviceVar = GetDevice(GPU);
            if (0 <= timeStep && timeStep < SequenceLength)
                return rDeviceVar != null ? rDeviceVar.DevicePointer + (timeStep * Count) * rDeviceVar.TypeSize : default(CUdeviceptr);
            return rDeviceVar != null ? rDeviceVar.DevicePointer + (TimeOffset + offset) * rDeviceVar.TypeSize : default(CUdeviceptr);
        }

        public override CUdeviceptr GetDevicePtr(MyAbstractObserver callee, int offset)
        {
            CudaDeviceVariable<T> rDeviceVar = GetDevice(MyKernelFactory.Instance.DevCount - 1);
            return rDeviceVar != null ? rDeviceVar.DevicePointer + (TimeOffset + offset) * rDeviceVar.TypeSize : default(CUdeviceptr);
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
