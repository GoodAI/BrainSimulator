using GoodAI.ToyWorld.Render;

namespace GoodAI.ToyWorld.Control
{
    public interface IGameController
    {
        void InitWorld();

        void Reset();

        void MakeStep();

        void AddRenderRequest(IRenderRequest renderRequest);

        IAvatarController GetAvatarController(int avatarId);
    }
}
