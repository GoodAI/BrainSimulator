using VRageMath;
using World.Physics;

namespace World.GameActors.GameObjects
{
    class Ball : Character
    {
        public Ball(
            string tilesetName,
            int tileId, string name,
            Vector2 position,
            Vector2 size,
            float direction) 
            : base(
            tilesetName,
            tileId,
            name,
            position,
            size,
            direction,
            typeof(CircleShape))
        {
        }
    }
}
