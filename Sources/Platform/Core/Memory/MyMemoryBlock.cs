using GoodAI.Core.Nodes;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

namespace GoodAI.Core.Memory
{
    public abstract class MyAbstractMemoryBlock
    {            
        public int Count { get; set; }        
        public string Name { get; set; }

        //TODO: Find if MyWorkingNode is possible here
        public MyNode Owner { get; set; }
        public int ColumnHint { get; set; }
        public TensorDimensions Dims { get; set; }
        public float MinValueHint { get; set; }
        public float MaxValueHint { get; set; }
                        
        public bool Persistable { get; internal set; }
        public bool Shared { get; protected set; }
        public bool IsOutput { get; internal set; }

        public bool Unmanaged { get; internal set; }
        public SizeT ExternalPointer { get; set; }

        public abstract void AllocateHost();
        public abstract void AllocateDevice();
        public abstract void FreeHost();
        public abstract void FreeDevice();
        public abstract bool SafeCopyToDevice();
        public abstract void SafeCopyToHost();
        public abstract CUdeviceptr GetDevicePtr(int GPU);
        public abstract CUdeviceptr GetDevicePtr(int GPU, int offset);
        public abstract CUdeviceptr GetDevicePtr(int GPU, int offset, int timeStep);
        public abstract CUdeviceptr GetDevicePtr(MyWorkingNode callee);
        public abstract CUdeviceptr GetDevicePtr(MyAbstractObserver callee);
        public abstract CUdeviceptr GetDevicePtr(MyWorkingNode callee, int offset);
        public abstract CUdeviceptr GetDevicePtr(MyAbstractObserver callee, int offset);
        public abstract SizeT GetSize();      
        public abstract void Synchronize();
        public abstract void GetBytes(byte[] destBuffer);
        public abstract void Fill(byte[] srcBuffer);

        public abstract void GetValueAt<T>(ref T value, int index);        
    }

    public class MyMemoryBlock<T> : MyAbstractMemoryBlock where T : struct
    {
        protected virtual CudaDeviceVariable<T>[] Device { get; set; }
        public T[] Host { get; protected set; }

        public bool OnDevice
        {
            get
            {
                return Device != null && Device[Owner.GPU] != null;
            }
        }

        public bool OnHost
        {
            get
            {
                return Host != null;
            }
        }

        public MyMemoryBlock()
        {
            Count = 0;
            ColumnHint = 1;
            MinValueHint = float.NegativeInfinity;
            MaxValueHint = float.PositiveInfinity;
            Dims = new TensorDimensions();
        }

        public override void AllocateHost()
        {
            Host = new T[Count];
        }

        public override void FreeHost()
        {
            Host = null;
        }

        public void AllocateMemory()
        {
            AllocateDevice();
            AllocateHost();
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
                        MyLog.DEBUG.WriteLine("Allocating: " + typeof(T).ToString() + ", " + Count * System.Runtime.InteropServices.Marshal.SizeOf(typeof(T)));
                        Device[Owner.GPU] = new CudaDeviceVariable<T>(
                           MyKernelFactory.Instance.GetContextByGPU(Owner.GPU).AllocateMemory(
                           Count * System.Runtime.InteropServices.Marshal.SizeOf(typeof(T))));

                        Fill(0);
                    }
                    else
                    {
                        if (ExternalPointer != 0)
                        {
                            Device[Owner.GPU] = new CudaDeviceVariable<T>(new CUdeviceptr(ExternalPointer), Count * sizeof(float));
                        }
                        else
                        {
                            throw new ArgumentOutOfRangeException("External Pointer not set for Unmanaged memory block.");
                        }
                    }
                }
            }
        }

        public override void FreeDevice()
        {
            if (OnDevice)
            {
                for (int i = 0; i < Device.Length; i++)
                {
                    if (Device[i] != null)
                    {
                        if (MyKernelFactory.Instance.IsContextAlive(i))
                        {
                            if (!Unmanaged || i != Owner.GPU)
                            {
                                MyKernelFactory.Instance.GetContextByGPU(i).FreeMemory(Device[i].DevicePointer);
                            }
                            Device[i].Dispose();
                        }
                        Device[i] = null;
                    }
                }
                Device = null;
                Shared = false;
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
                Device[Owner.GPU].CopyToDevice(Host);
                return true;
            }
            else return false;
        }

        public virtual bool SafeCopyToDevice(int offset, int length)
        {
            if (!OnDevice)
            {
                AllocateDevice();
            }

            if (OnDevice)
            {
                int size = Marshal.SizeOf(typeof(T));
                Device[Owner.GPU].CopyToDevice(Host, size * offset, size * offset, size * length);
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
                Device[Owner.GPU].CopyToHost(Host);
            }
        }

        public virtual void SafeCopyToHost(int offset, int length)
        {
            if (!OnHost)
            {
                AllocateHost();
            }

            if (OnDevice)
            {
                int size = Marshal.SizeOf(typeof(T));
                Device[Owner.GPU].CopyToHost(Host, size * offset, size * offset, size * length);
            }
        }

        public virtual void CopyFromMemoryBlock(MyMemoryBlock<T> source, int srcOffset, int destOffset, int count)
        {
            int size = Marshal.SizeOf(typeof(T));
            Device[Owner.GPU].CopyToDevice(source.GetDevice(Owner.GPU), srcOffset * size, destOffset * size, count * size);
        }

        public virtual void CopyToMemoryBlock(MyMemoryBlock<T> destination, int srcOffset, int destOffset, int count)
        {
            int size = Marshal.SizeOf(typeof(T));
            destination.GetDevice(Owner.GPU).CopyToDevice(Device[Owner.GPU], srcOffset * size, destOffset * size, count * size);
        }

        protected void CopyToGPU(int nGPU)
        {
            if (Device[nGPU] != null)
            {
                CudaContext rContextDest = MyKernelFactory.Instance.GetContextByGPU(nGPU);
                CudaContext rContextSrc = MyKernelFactory.Instance.GetContextByGPU(Owner.GPU);
                Device[nGPU].PeerCopyToDevice(rContextDest, Device[Owner.GPU].DevicePointer, rContextSrc);
            }
        }

        public virtual CudaDeviceVariable<T> GetDevice(MyWorkingNode callee)
        {
            return GetDevice(callee.GPU);
        }

        public virtual CudaDeviceVariable<T> GetDevice(int nGPU)
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
                            Count * Marshal.SizeOf(typeof(T))));

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
            CudaDeviceVariable<T> rDeviceVar = GetDevice(GPU);
            return rDeviceVar != null ? rDeviceVar.DevicePointer + offset * rDeviceVar.TypeSize : default(CUdeviceptr);
        }

        public override CUdeviceptr GetDevicePtr(int GPU, int offset, int timestep)
        {
            return GetDevicePtr(GPU, offset);
        }

        public override CUdeviceptr GetDevicePtr(MyWorkingNode callee)
        {
            return GetDevicePtr(callee, 0);
        }

        public override CUdeviceptr GetDevicePtr(MyWorkingNode callee, int offset)
        {
            return GetDevicePtr(callee.GPU, offset);
        }

        public override CUdeviceptr GetDevicePtr(MyAbstractObserver callee)
        {
            return GetDevicePtr(callee, 0);
        }

        public override CUdeviceptr GetDevicePtr(MyAbstractObserver callee, int offset)
        {
            CudaDeviceVariable<T> rDeviceVar = GetDevice(MyKernelFactory.Instance.DevCount - 1);
            return rDeviceVar != null ? rDeviceVar.DevicePointer + offset * rDeviceVar.TypeSize : default(CUdeviceptr);
        }

        public override SizeT GetSize()
        {
            return Count * Marshal.SizeOf(typeof(T));
        }

        public override void Synchronize()
        {
            if (Shared)
            {
                for (int i = 0; i < MyKernelFactory.Instance.DevCount; i++)
                {
                    if (i != Owner.GPU)
                    {
                        CopyToGPU(i);
                    }
                }
            }
        }

        public virtual void Fill(float value)
        {
            Device[Owner.GPU].Memset(BitConverter.ToUInt32(BitConverter.GetBytes(value), 0));
        }

        public virtual void Fill(int value)
        {
            Device[Owner.GPU].Memset(BitConverter.ToUInt32(BitConverter.GetBytes(value), 0));
        }

        public virtual void Fill(uint value)
        {
            Device[Owner.GPU].Memset(BitConverter.ToUInt32(BitConverter.GetBytes(value), 0));
        }

        public virtual void Fill(bool value)
        {
            Device[Owner.GPU].Memset(BitConverter.ToUInt32(BitConverter.GetBytes(value), 0));
        }

        public override void Fill(byte[] srcBuffer)
        {
            SizeT size = GetSize();
            if (size > srcBuffer.Length) throw new ArgumentException("Source buffer to small (" + size + "->" + srcBuffer.Length + ")");

            Buffer.BlockCopy(srcBuffer, 0, Host, 0, size);
            SafeCopyToDevice();
        }

        public override void GetBytes(byte[] destBuffer)
        {
            SizeT size = GetSize();
            if (size > destBuffer.Length) throw new ArgumentException("Destinantion buffer to small (" + size + "->" + destBuffer.Length + ")");

            SafeCopyToHost();
            Buffer.BlockCopy(Host, 0, destBuffer, 0, size);
        }

        public virtual T GetValueAt(int index)
        {
            T value = new T();                

            if (OnDevice && index < Count)
            {            
                Device[Owner.GPU].CopyToHost(ref value, index * Marshal.SizeOf(typeof(T)));            
            }

            return value;
        }

        public override void GetValueAt<TR>(ref TR value, int index)
        {
            TypeConverter tc = TypeDescriptor.GetConverter(typeof(T));

            if (tc.CanConvertTo(typeof(TR)))
            {
                value = (TR)tc.ConvertTo(GetValueAt(index), typeof(TR));
            }
        }
    }
}
