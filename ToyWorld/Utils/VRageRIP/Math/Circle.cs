using VRageMath;

namespace VRageMath
{
    public struct Circle
    {
        public Vector2 Center;
        public float Radius;

        public Circle(Vector2 center, float radius)
        {
            Center = center;
            Radius = radius;
        }

        public bool Include(Vector2 point)
        {
            return Vector2.Distance(Center, point) <= Radius;
        }
    }
}