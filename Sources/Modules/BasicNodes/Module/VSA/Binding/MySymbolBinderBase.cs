using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using ManagedCuda.BasicTypes;
using System.Collections.Generic;
using System.Linq;

namespace GoodAI.Modules.VSA
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


        public abstract void Bind(CUdeviceptr firstInput, IEnumerable<CUdeviceptr> otherInputs, CUdeviceptr output);

        public virtual void Bind(CUdeviceptr firstInput, CUdeviceptr otherInput, CUdeviceptr output)
        {
            Bind(firstInput, Enumerable.Repeat(otherInput, 1), output);
        }

        public virtual void Bind(MyMemoryBlock<float> firstInput, MyMemoryBlock<float> secondInput, MyMemoryBlock<float> output)
        {
            int nrInputs = secondInput.Count / m_inputSize;

            var vecs = nrInputs > 1
                // Concatenate pointers to the individual vectors
                ? Enumerable.Range(0, nrInputs).Select(i => secondInput.GetDevicePtr(m_owner) + i * m_inputSize * sizeof(float))
                // Use only a singe pointer
                : Enumerable.Repeat(secondInput.GetDevicePtr(m_owner), 1);

            Bind(firstInput.GetDevicePtr(m_owner), vecs, output.GetDevicePtr(m_owner));
        }


        public abstract void Unbind(CUdeviceptr firstInput, IEnumerable<CUdeviceptr> otherInputs, CUdeviceptr output);

        public virtual void Unbind(CUdeviceptr firstInput, CUdeviceptr otherInput, CUdeviceptr output)
        {
            Unbind(firstInput, Enumerable.Repeat(otherInput, 1), output);
        }

        public virtual void Unbind(MyMemoryBlock<float> firstInput, MyMemoryBlock<float> secondInput, MyMemoryBlock<float> output)
        {
            int nrInputs = secondInput.Count / m_inputSize;

            var vecs = nrInputs > 1
                // Concatenate pointers to the individual vectors
                ? Enumerable.Range(0, nrInputs).Select(i => secondInput.GetDevicePtr(m_owner) + i * m_inputSize * sizeof(float))
                // Use only a singe pointer
                : Enumerable.Repeat(secondInput.GetDevicePtr(m_owner), 1);

            Unbind(firstInput.GetDevicePtr(m_owner), vecs, output.GetDevicePtr(m_owner));
        }
    }
}
