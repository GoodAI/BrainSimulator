using GoodAI.Core.Memory;
using System;
using System.Linq;

namespace MNIST
{
    public interface IExample
    {
        float[] Input { get; }
        int Target { get; }
    }

    public class NormalizedExample : IExample
    {
        private float[] m_input;
        private int m_target;

        public float[] Input { get { return m_input; } }
        public int Target { get { return m_target; } }

        public NormalizedExample(float[] input, int target)
        {
            float min = float.MaxValue;
            float max = float.MinValue;
            m_input = input;
            m_target = target;


            for (int i = 0; i < input.Length; i++)
            {
                if (max < m_input[i]) max = m_input[i];
                if (min > m_input[i]) min = m_input[i];
            }

            float divBy = max - min;

            for (int i = 0; i < input.Length; i++)
            {
                m_input[i] = (m_input[i] - min) / divBy;
            }
        }

        public NormalizedExample(byte[] input, int target)
        // slower by 0.3 sec for MNIST dataset
        //: this(Array.ConvertAll(input, v => (float) v), target)
        //{ }
        {
            byte min = byte.MaxValue;
            byte max = byte.MinValue;
            m_input = new float[input.Length];
            m_target = target;


            for (int i = 0; i < input.Length; i++)
            {
                if (max < input[i]) max = input[i];
                if (min > input[i]) min = input[i];
            }

            float divBy = max - min;

            for (int i = 0; i < input.Length; i++)
            {
                m_input[i] = (input[i] - min) / divBy;
            }
        }
    }
}
