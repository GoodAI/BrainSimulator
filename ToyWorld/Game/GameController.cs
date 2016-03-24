using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Control;
using GoodAI.ToyWorld.Render;
using GoodAI.ToyWorld.Render.RenderRequests;
using Render.Renderer;
using Render.RenderRequests;
using Render.RenderRequests.Setup;

namespace Game
{
    public class TestGameController : IGameController
    {
        private readonly IRenderer m_renderer = new GLRenderer();


        #region IGameController overrides

        public void InitWorld()
        { }

        public void Reset()
        { }

        public void MakeStep()
        { }

        public T RegisterAgentRenderRequest<T>(int agentID)
            where T : IAgentRenderRequest
        {
            throw new NotImplementedException();
        }

        public T RegisterRenderRequest<T>()
            where T : IRenderRequest
        {
            var rr = RenderRequestFactory.CreateRenderRequest<T>();
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
