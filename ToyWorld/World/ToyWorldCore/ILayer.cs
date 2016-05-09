using VRageMath;
using World.GameActors;

namespace World.ToyWorldCore
{
    public interface ILayer<out T> where T : GameActor
    {
        LayerType LayerType { get; set; }

        T GetActorAt(int x, int y);

        T GetActorAt(Vector2I position);

        bool ReplaceWith<TR>(GameActorPosition original, TR replacement);

        bool Add(GameActorPosition gameActorPosition);
    }
}
