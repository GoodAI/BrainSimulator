using System;
namespace ToyWorldTests
{
    class TestUtils
    {
        public static bool FloatEq(float f1, float f2, float eps = 3.0517578E-5f)
        {
            return Math.Abs(f1 - f2) < eps;
        }
    }
}