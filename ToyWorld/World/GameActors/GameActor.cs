using VRageMath;
using World.Atlas.Layers;
using World.ToyWorldCore;

namespace World.GameActors
{
    public interface IGameActor
    {
        int TilesetId { get; set; }
    }

    /// <summary>
    /// Common ancestor of GameObjects and Tiles
    /// </summary>
    public abstract class GameActor : IGameActor
    {
        /// <summary>
        /// Serial number of texture in tileset.
        /// </summary>
        public int TilesetId { get; set; }
    }

    public class GameActorPosition
    {
        public GameActor Actor { get; private set; }
        public Vector2 Position { get; private set; }
        public LayerType Layer { get; private set; }

        public GameActorPosition(GameActor actor, Vector2 position, LayerType layer)
        {
            Actor = actor;
            Position = position;
            Layer = layer;
        }
    }
}