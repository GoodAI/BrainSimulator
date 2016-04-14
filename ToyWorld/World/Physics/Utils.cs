using System;
using VRageMath;

namespace World.Physics
{
    class Utils
    {
        public static Vector2 Move(Vector2 initialPosition, float directionInDegrees, float speed)
        {
            var directionInRads = VRageMath.MathHelper.ToRadians(directionInDegrees);
            var cos = (float)Math.Cos(directionInRads);
            float x = initialPosition.X + cos * speed;
            var sin = (float)Math.Sin(directionInRads);
            float y = initialPosition.Y + sin * speed;
            return new Vector2(x, y);
        }
    }
}
