using System;
using VRageMath;
using World.WorldInterfaces;
using Xunit;


namespace ToyWorldTests.Physics
{
    public class UtilsTest
    {
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
                    Assert.Equal(decomposeSpeed.Y, MathHelper.Sqrt2/2, 2);
                    break;
                case -135:
                    Assert.Equal(decomposeSpeed.X, MathHelper.Sqrt2/2, 2);
                    Assert.Equal(decomposeSpeed.Y, -MathHelper.Sqrt2/2, 2);
                    break;
                case 90:
                    Assert.Equal(decomposeSpeed.X, -2, 2);
                    Assert.Equal(decomposeSpeed.Y, 0, 2);
                    break;
            }
        }

        [Theory]
        [InlineData(0, 1)]
        [InlineData(45, 1)]
        [InlineData(-135, 2)]
        [InlineData(90, 1)]
        public void TestDecomposeSpeedWithRefDir(float direction, float speed)
        {
            float referenceDirection = MathHelper.ToRadians(45);
            float directionRads = MathHelper.ToRadians(direction);
            Vector2 decomposeSpeed = global::World.Physics.Utils.DecomposeSpeed(speed, directionRads, referenceDirection);

            switch ((int) direction)
            {
                case 0:
                    Assert.Equal(decomposeSpeed.X, MathHelper.Sqrt2/2, 2);
                    Assert.Equal(decomposeSpeed.Y, MathHelper.Sqrt2/2, 2);
                    break;
                case 45:
                    Assert.Equal(decomposeSpeed.X, 0, 2);
                    Assert.Equal(decomposeSpeed.Y, 1, 2);
                    break;
                case -135:
                    Assert.Equal(decomposeSpeed.X, 0, 2);
                    Assert.Equal(decomposeSpeed.Y, -2, 2);
                    break;
                case 90:
                    Assert.Equal(decomposeSpeed.X, -MathHelper.Sqrt2/2, 2);
                    Assert.Equal(decomposeSpeed.Y, MathHelper.Sqrt2/2, 2);
                    break;
            }
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