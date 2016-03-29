using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Control;
using Render.Renderer;
using Render.RenderRequests;

namespace Game
{
    public abstract class GameControllerBase : IGameController
    {
        protected IRenderer Renderer { get; private set; }
        //protected IWorld World { get; private set; }


        protected GameControllerBase(IRenderer renderer)
        {
            Renderer = renderer;
        }


        #region IGameController overrides

        public virtual void InitWorld(GameSetup setup)
        { }

        public virtual void Reset()
        { }

        public virtual void MakeStep()
        {
            //if (World == null)
            //    throw new 
        }


        public virtual T RegisterRenderRequest<T>()
            where T : class, IRenderRequest
        {
            var rr = RenderRequestFactory.CreateRenderRequest<T>();
            Renderer.EnqueueRequest(rr);

            return rr;
        }

        public virtual T RegisterAgentRenderRequest<T>(int avatarID)
            where T : class, IAgentRenderRequest
        {
            // TODO: check agentID or make the param an AgentController?

            var rr = RenderRequestFactory.CreateAgentRenderRequest<T>(avatarID);
            Renderer.EnqueueRequest(rr);

            return rr;
        }


        public virtual IAvatarController GetAvatarController(int avatarId)
        {
            return null;
        }

        #endregion
    }
}
