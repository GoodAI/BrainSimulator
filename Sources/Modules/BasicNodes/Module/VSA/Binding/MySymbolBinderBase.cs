using BrainSimulator.Memory;
using BrainSimulator.Nodes;
using BrainSimulator.Transforms;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaFFT;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BrainSimulator.VSA
{
    public abstract class MySymbolBinderBase
    {
        private readonly Dictionary<int, CUdeviceptr[]> tempArrays = new Dictionary<int, CUdeviceptr[]>();

        protected int m_inputSize;
        protected MyMemoryBlock<float> m_tempBlock;
        protected MyWorkingNode m_owner;

        public bool NormalizeOutput { get; set; }
        public float Denominator { get; set; }
        public bool ExactQuery { get; set; }


        public MySymbolBinderBase(MyWorkingNode owner, int inputSize, MyMemoryBlock<float> tempBlock)
        {
            m_inputSize = inputSize;
            m_tempBlock = tempBlock;
            m_owner = owner;
        }


        protected CUdeviceptr[] GetTempArray(int length)
        {
            CUdeviceptr[] temp;

            tempArrays.TryGetValue(length, out temp);
            if (temp == null)
            {
                temp = new CUdeviceptr[length];
                tempArrays.Add(length, temp);
            }

            return temp;
        }


        public abstract void Bind(CUdeviceptr firstInput, params CUdeviceptr[] otherInputs);

        public virtual void Bind(MyMemoryBlock<float> firstInput, MyMemoryBlock<float> secondInput, MyMemoryBlock<float> output)
        {
            Bind(firstInput.GetDevicePtr(m_owner), secondInput.GetDevicePtr(m_owner), output.GetDevicePtr(m_owner));
        }

        public virtual void Bind(MyMemoryBlock<float> inputs, MyMemoryBlock<float> output)
        {
            int nrInputs = inputs.Count / m_inputSize;
            CUdeviceptr start = inputs.GetDevicePtr(m_owner);
            CUdeviceptr[] arr = GetTempArray(nrInputs); //-1 to skip the first +1 to include output
            for (int i = 0; i < nrInputs - 1; ++i)
            {
                arr[i] = start + (i + 1) * m_inputSize * sizeof(float);
            }

            arr[nrInputs - 1] = output.GetDevicePtr(m_owner);

            Bind(start, arr);
        }


        public abstract void Unbind(CUdeviceptr firstInput, params CUdeviceptr[] otherInputs);

        public virtual void Unbind(MyMemoryBlock<float> firstInput, MyMemoryBlock<float> secondInput, MyMemoryBlock<float> output)
        {
            Unbind(firstInput.GetDevicePtr(m_owner), secondInput.GetDevicePtr(m_owner), output.GetDevicePtr(m_owner));
        }

        public virtual void UnbindMultiple(MyMemoryBlock<float> firstInput, MyMemoryBlock<float> otherInputs, MyMemoryBlock<float> output)
        {
            int nrInputs = otherInputs.Count / m_inputSize;
            CUdeviceptr firstPtr = firstInput.GetDevicePtr(m_owner);
            CUdeviceptr start = otherInputs.GetDevicePtr(m_owner);
            CUdeviceptr[] arr = GetTempArray(nrInputs + 1);//+1 for output

            for (int i = 0; i <= nrInputs; ++i)
            {
                arr[i] = start + i * m_inputSize * sizeof(float);
            }

            arr[nrInputs] = output.GetDevicePtr(m_owner);

            Unbind(firstPtr, arr);
        }
    }
}
