using System;
using VRageMath;
using Xunit;


namespace ToyWorldTests.Physics
{
    public class UtilsTest
    {
        private const float HALF_SQRT2_F = MathHelper.Sqrt2 / 2;

        [Theory]
        [InlineData(0, 1)]
        [InlineData(45, 1)]
        [InlineData(-135, 1)]
        [InlineData(90, 2)]
        public void TestDecomposeSpeed(float direction, float speed)
        {
            float directionRads = MathHelper.ToRadians(direction);
            Vector2 decomposeSpeed = global::World.Physics.Utils.DecomposeSpeed(speed, directionRads);

            switch ((int) direction)
            {
                case 0:
                    Assert.Equal(decomposeSpeed.X, 0, 3);
                    Assert.Equal(decomposeSpeed.Y, speed, 3);
                    break;
                case 45:
                    Assert.Equal(decomposeSpeed.X, -MathHelper.Sqrt2/2, 2);
                    Assert.Equal(decomposeSpeed.Y, HALF_SQRT2_F, 2);
                    break;
                case -135:
                    Assert.Equal(decomposeSpeed.X, HALF_SQRT2_F, 2);
                    Assert.Equal(decomposeSpeed.Y, -MathHelper.Sqrt2/2, 2);
                    break;
                case 90:
                    Assert.Equal(decomposeSpeed.X, -2, 2);
                    Assert.Equal(decomposeSpeed.Y, 0, 2);
                    break;
            }
        }

        [Theory]
        [InlineData(0, 1, HALF_SQRT2_F, HALF_SQRT2_F)]
        [InlineData(45, 1, 0, 1)]
        [InlineData(-135, 2, 0, -2)]
        [InlineData(90, 1, -HALF_SQRT2_F, HALF_SQRT2_F)]
        public void TestDecomposeSpeedWithRefDir(float direction, float speed, float expectedX, float expectedY)
        {
            float referenceDirection = MathHelper.ToRadians(45);
            float directionRads = MathHelper.ToRadians(direction);
            Vector2 decomposeSpeed = global::World.Physics.Utils.DecomposeSpeed(speed, directionRads, referenceDirection);

            Assert.Equal(decomposeSpeed.X, expectedX, 2);
            Assert.Equal(decomposeSpeed.Y, expectedY, 2);
        }

        [Theory]
        [InlineData(0, 1)]
        [InlineData(45, 1)]
        [InlineData(-135, 2)]
        [InlineData(90, 1)]
        public void TestComposeSpeed(float direction, float speed)
        {
            float directionRads = MathHelper.ToRadians(direction);
            Vector2 decomposeSpeed = global::World.Physics.Utils.DecomposeSpeed(speed, directionRads);
            Tuple<float, float> composeSpeed = global::World.Physics.Utils.ComposeSpeed(decomposeSpeed);
            Assert.Equal(composeSpeed.Item1, speed, 3);
            Assert.Equal(composeSpeed.Item2, directionRads, 3);
        }

        [Theory]
        [InlineData(0, 1)]
        [InlineData(45, 1)]
        [InlineData(-135, 2)]
        [InlineData(90, 1)]
        public void TestComposeSpeedWithRefDir(float direction, float speed)
        {
            float referenceDirection = MathHelper.ToRadians(45);
            float directionRads = MathHelper.ToRadians(direction);
            Vector2 decomposeSpeed = global::World.Physics.Utils.DecomposeSpeed(speed, directionRads, referenceDirection);
            Tuple<float, float> composeSpeed = global::World.Physics.Utils.ComposeSpeed(decomposeSpeed,
                referenceDirection);
            Assert.Equal(composeSpeed.Item1, speed, 3);
            Assert.Equal(composeSpeed.Item2, directionRads, 3);
        }

        [Theory]
        [InlineData(0, 1)]
        [InlineData(45, 1)]
        [InlineData(-135, 2)]
        [InlineData(90, 1)]
        public void TestComposeSpeedWithRefDir2(float direction, float speed)
        {
            float referenceDirection = MathHelper.ToRadians(-45);
            float directionRads = MathHelper.ToRadians(direction);
            Vector2 decomposeSpeed = global::World.Physics.Utils.DecomposeSpeed(speed, directionRads, referenceDirection);
            Tuple<float, float> composeSpeed = global::World.Physics.Utils.ComposeSpeed(decomposeSpeed,
                referenceDirection);
            Assert.Equal(composeSpeed.Item1, speed, 3);
            Assert.Equal(composeSpeed.Item2, directionRads, 3);
        }
    }
}