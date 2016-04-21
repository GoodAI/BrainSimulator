using System;
using VRageMath;

namespace World.Physics
{
    class Utils
    {
        public static Vector2 Move(Vector2 initialPosition, float directionInRads, float speed)
        {
            Vector2 direction = new Vector2(-(float)Math.Sin(directionInRads), (float)Math.Cos(directionInRads));
            return initialPosition + direction * speed;
        }
    }
}
