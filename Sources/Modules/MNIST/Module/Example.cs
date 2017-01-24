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
        public float[] Input { get; }
        public int Target { get; }

        public NormalizedExample(float[] input, int target)
        {
            float min = float.MaxValue;
            float max = float.MinValue;
            Input = input;
            Target = target;


            for (int i = 0; i < input.Length; i++)
            {
                if (max < Input[i]) max = Input[i];
                if (min > Input[i]) min = Input[i];
            }

            float divBy = max - min;

            for (int i = 0; i < input.Length; i++)
            {
                Input[i] = (Input[i] - min) / divBy;
            }
        }

        public NormalizedExample(byte[] input, int target)
        // slower by 0.3 sec for MNIST dataset
        //: this(Array.ConvertAll(input, v => (float) v), target)
        //{ }
        {
            byte min = byte.MaxValue;
            byte max = byte.MinValue;
            Input = new float[input.Length];
            Target = target;


            for (int i = 0; i < input.Length; i++)
            {
                if (max < input[i]) max = input[i];
                if (min > input[i]) min = input[i];
            }

            float divBy = max - min;

            for (int i = 0; i < input.Length; i++)
            {
                Input[i] = (input[i] - min) / divBy;
            }
        }
    }
}
