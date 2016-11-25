using System;
using VRageMath;

namespace World.Physics
{
    public class Utils
    {
        public static Vector2 Move(Vector2 initialPosition, float directionInRads, float speed)
        {
            Vector2 direction = new Vector2(-(float)Math.Sin(directionInRads), (float)Math.Cos(directionInRads));
            return initialPosition + direction * speed;
        }

        public static Vector2 DecomposeSpeed(float speed, float direction)
        {
            return new Vector2(-speed * (float)Math.Sin(direction), speed * (float)Math.Cos(direction));
        }

        public static Vector2 DecomposeSpeed(float speed, float direction, float referenceDirection)
        {
            return new Vector2(
                -speed*(float) Math.Sin(direction - referenceDirection),
                speed*(float) Math.Cos(direction - referenceDirection));
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="speeds"></param>
        /// <returns>Speed, Direction.</returns>
        public static Tuple<float, float> ComposeSpeed(Vector2 speeds)
        {
            return new Tuple<float, float>(speeds.Length(), MathHelper.WrapAngle((float)Math.Atan2(-speeds.X, speeds.Y)));
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="speeds"></param>
        /// <param name="referenceDirection"></param>
        /// <returns>Speed, Direction.</returns>
        public static Tuple<float, float> ComposeSpeed(Vector2 speeds, float referenceDirection)
        {
            return new Tuple<float, float>(speeds.Length(), MathHelper.WrapAngle((float)Math.Atan2(-speeds.X,speeds.Y) + referenceDirection));
        }
    }
}
