using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.ToyWorld.Language
{
    /// <summary>
    /// Vectors with string labels and basic vector operations
    /// </summary>
    public class LabeledVector
    {
        public string Label { get; private set; }

        public float[] Vector { get; private set; }

        public float Length
        {
            get
            {
                return VectorLength(Vector);
            }
        }

        public static float VectorLength(float[] vector)
        {
            return (float)Math.Sqrt(InnerProduct(vector, vector));
        }

        public LabeledVector(string label, float[] vector)
        {
            Label = label;
            Vector = vector;
        }

        public float InnerProduct(LabeledVector otherVector)
        {
            return InnerProduct(otherVector.Vector);
        }

        public float InnerProduct(float[] otherVector)
        {
            // TODO check dimensions of other vector
            int numberOfDimensions = Vector.Count();
            float product = 0;
            for (int elementIndex = 0; elementIndex < numberOfDimensions; elementIndex++)
            {
                product += Vector[elementIndex] * otherVector[elementIndex];
            }
            return product;
        }

        public static float InnerProduct(float[] vector1, float[] vector2)
        {
            int numberOfDimensions = vector1.Count();
            float product = 0;
            for (int elementIndex = 0; elementIndex < numberOfDimensions; elementIndex++)
            {
                product += vector1[elementIndex] * vector2[elementIndex];
            }
            return product;
        }

        public float Cosine(LabeledVector otherVector)
        {
            return Cosine(otherVector.Vector);
        }

        public float Cosine(float[] otherVector)
        {
            return Cosine(Vector, otherVector);
        }

        public static float Cosine(float[] vector1, float[] vector2)
        {
            return InnerProduct(vector1, vector2) / (VectorLength(vector1) * VectorLength(vector2));
        }

        public float Euclidean(LabeledVector otherVector)
        {
            return Euclidean(otherVector.Vector);
        }

        public float Euclidean(float[] otherVector)
        {
            return Euclidean(Vector, otherVector);
        }

        public static float Euclidean(float[] vector1, float[] vector2)
        {
            return (float)Math.Sqrt(SquaredEuclidean(vector1, vector2));
        }

        public static float SquaredEuclidean(float[] vector1, float[] vector2)
        {
            int numberOfDimensions = vector1.Count();
            float sumOfSquares = 0;
            for (int elementIndex = 0; elementIndex < numberOfDimensions; elementIndex++)
            {
                float difference = vector1[elementIndex] - vector2[elementIndex];
                sumOfSquares += difference * difference;
            }
            return sumOfSquares;
        }

        /// <summary>
        /// Performs elementwise addition of vectors.
        /// </summary>
        /// <param name="vector1">a vector</param>
        /// <param name="vector2">another vector</param>
        /// <returns>the sum of the vectors</returns>
        public static float[] Add(float[] vector1, float[] vector2)
        {
            int length = vector1.Count();
            if (length != vector2.Count())
            {
                throw new ArgumentException("Cannot add vectors of different lengths");
            }
            float[] result = new float[length];
            for (int elementIndex = 0; elementIndex < length; elementIndex++)
            {
                result[elementIndex] = vector1[elementIndex] + vector2[elementIndex];
            }
            return result;
        }

        /// <summary>
        /// Performs elementwise subtraction of vectors.
        /// </summary>
        /// <param name="vector1">first vector</param>
        /// <param name="vector2">second vector</param>
        /// <returns>the difference between the vectors</returns>
        public static float[] Subtract(float[] vector1, float[] vector2)
        {
            int length = vector1.Count();
            if (length != vector2.Count())
            {
                throw new ArgumentException("Cannot subtract vectors of different lengths");
            }
            float[] result = new float[length];
            for (int elementIndex = 0; elementIndex < length; elementIndex++)
            {
                result[elementIndex] = vector1[elementIndex] - vector2[elementIndex];
            }
            return result;
        }
    }
}
