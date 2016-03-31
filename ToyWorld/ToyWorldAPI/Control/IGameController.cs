using System;

namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public interface IGameController : IDisposable
    {
        /// <summary>
        /// 
        /// </summary>
        void Init(GameSetup setup);

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
        /// <param name="avatarID"></param>
        /// <exception cref="RenderRequestNotImplementedException">Thrown when requesting an unknown <see cref="IAvatarRenderRequest"/> from the controller.
        /// This usually indicates an older version of the core than the API.</exception>
        T RegisterAvatarRenderRequest<T>(int avatarID)
            where T : class, IAvatarRenderRequest;

        /// <summary>
        /// 
        /// </summary>
        /// <exception cref="RenderRequestNotImplementedException">Thrown when requesting an unknown <see cref="IRenderRequest"/> from the controller.
        /// This usually indicates an older version of the core than the API.</exception>
        T RegisterRenderRequest<T>()
            where T : class, IRenderRequest;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="avatarId"></param>
        /// <returns></returns>
        IAvatarController GetAvatarController(int avatarId);
    }
}
