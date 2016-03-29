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
        /// <param name="agentID"></param>
        T RegisterAgentRenderRequest<T>(int agentID)
            where T : class, IAgentRenderRequest;

        /// <summary>
        /// 
        /// </summary>
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
