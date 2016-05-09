using VRageMath;
using World.ToyWorldCore;

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
        public Vector2 Position { get; private set; }
        public LayerType Layer { get; private set; }

        public GameActorPosition(GameActor actor, Vector2 position, LayerType layer = LayerType.ObstacleInteractable)
        {
            Actor = actor;
            Position = position;
            Layer = layer;
        }
    }
}