using VRageMath;

namespace World.GameActors
{
    /// <summary>
    /// Common ancestor of GameObjects and Tiles
    /// </summary>
    public abstract class GameActor
    {
    }

    public class GameActorPosition
    {
        public GameActor Actor { get; private set; }
        public Vector2I Position { get; private set; }

        public GameActorPosition(GameActor actor, Vector2I position)
        {
            Actor = actor;
            Position = position;
        }
    }
}