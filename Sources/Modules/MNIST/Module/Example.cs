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

    public class Example : IExample
    {
        private float[] _input;
        private int _target;

        public float[] Input { get { return _input; } }

        public int Target { get { return _target; } }


        public Example(float[] input, int target)
        {
            _input = input;
            _target = target;
        }

        public Example(byte[] input, int target)
            : this(Array.ConvertAll(input, v => (float) v), target)
        {
        }
    }
}
