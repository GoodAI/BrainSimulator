using GoodAI.ToyWorld.Render;

namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public interface IGameController
    {
        /// <summary>
        /// 
        /// </summary>
        void InitWorld();

        /// <summary>
        /// 
        /// </summary>
        void Reset();

        /// <summary>
        /// 
        /// </summary>
        void MakeStep();

        /// <summary>
        /// 
        /// </summary>
        /// <param name="renderRequest"></param>
        void AddRenderRequest(IRenderRequest renderRequest);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="avatarId"></param>
        /// <returns></returns>
        IAvatarController GetAvatarController(int avatarId);
    }
}
