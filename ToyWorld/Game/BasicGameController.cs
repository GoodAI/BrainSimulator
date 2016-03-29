using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Control;
using Render.Renderer;
using Render.RenderRequests;
using Render.RenderRequests.Setup;

namespace Game
{
    public class BasicGameController : IGameController
    {
        private readonly IRenderer m_renderer = new GLRenderer();


        #region IGameController overrides

        public void InitWorld()
        { }

        public void Reset()
        { }

        public void MakeStep()
        { }

        public T RegisterRenderRequest<T>()
            where T : class, IRenderRequest
        {
            var rr = RenderRequestFactory.CreateRenderRequest<T>();
            m_renderer.EnqueueRequest(rr);

            return rr;
        }

        public T RegisterAgentRenderRequest<T>(int agentID)
            where T : class, IAgentRenderRequest
        {
            var rr = RenderRequestFactory.CreateAgentRenderRequest<T>();
            m_renderer.EnqueueRequest(rr);

            return rr;
        }

        public IAvatarController GetAvatarController(int avatarId)
        {
            throw new NotImplementedException();
        }

        #endregion
    }
}
