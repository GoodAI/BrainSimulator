using GoodAI.ToyWorld.Control;

namespace Physics
{
    public class MovementPhysics : IPhysics
    {
        public static void Move(IMovable movable)
        {
            Shift(movable);
            Rotate(movable);
        }

        private static void Shift(IMovable movable)
        {
            throw new System.NotImplementedException();
        }

        private static void Rotate(IMovable movable)
        {
            throw new System.NotImplementedException();
        }
    }
}
