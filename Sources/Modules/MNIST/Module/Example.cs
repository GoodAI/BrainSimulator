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
        private float[] _input;
        private int _target;

        public float[] Input { get { return _input; } }
        public int Target { get { return _target; } }

        public NormalizedExample(float[] input, int target)
        {
            float min = float.MaxValue;
            float max = float.MinValue;
            _input = input;
            _target = target;


            for (int i = 0; i < input.Length; i++)
            {
                if (max < _input[i]) max = _input[i];
                if (min > _input[i]) min = _input[i];
            }

            float divBy = max - min;

            for (int i = 0; i < input.Length; i++)
            {
                _input[i] = (_input[i] - min) / divBy;
            }
        }

        public NormalizedExample(byte[] input, int target)
        // slower by 0.3 sec for MNIST dataset
        //: this(Array.ConvertAll(input, v => (float) v), target)
        //{ }
        {
            byte min = byte.MaxValue;
            byte max = byte.MinValue;
            _input = new float[input.Length];
            _target = target;


            for (int i = 0; i < input.Length; i++)
            {
                if (max < input[i]) max = input[i];
                if (min > input[i]) min = input[i];
            }

            float divBy = max - min;

            for (int i = 0; i < input.Length; i++)
            {
                _input[i] = (input[i] - min) / divBy;
            }
        }
    }
}
