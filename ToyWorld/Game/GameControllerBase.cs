using System;
using System.Collections.Generic;
using System.Diagnostics;
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
        public IRenderer Renderer { get; private set; }
        //protected IWorld World { get; private set; }


        protected GameControllerBase(IRenderer renderer)
        {
            Renderer = renderer;
        }

        public virtual void Dispose()
        {
            if (Renderer != null)
                Renderer.Dispose();
            Renderer = null;
        }


        #region IGameController overrides

        public virtual void Init(GameSetup setup)
        {
            // TODO: world

            Renderer.Init();
            Renderer.CreateWindow("TestGameWindow", 1024, 1024);
            Renderer.CreateContext();
        }

        public virtual void Reset()
        {
            // TODO: Semantics of Reset? What should it do?
        }


        public virtual void MakeStep()
        {
            // Assume Init has been called, we don't want to check for consistency every step

            Renderer.ProcessRequests();
        }


        public virtual T RegisterRenderRequest<T>()
            where T : class, IRenderRequest
        {
            var rr = RenderRequestFactory.CreateRenderRequest<T>();
            InitRR(rr);
            Renderer.EnqueueRequest(rr);

            return rr;
        }

        public virtual T RegisterAvatarRenderRequest<T>(int avatarID)
            where T : class, IAvatarRenderRequest
        {
            // TODO: check agentID or make the param an AgentController?

            var rr = RenderRequestFactory.CreateAvatarRenderRequest<T>(avatarID);
            InitRR(rr);
            Renderer.EnqueueRequest(rr);

            return rr;
        }

        void InitRR<T>(T rr)
            where T : class
        {
            var rrBase = rr as RenderRequest; // Assume that all renderRequests created by factory inherit from RenderRequest

            if (rrBase == null)
                throw new RenderRequestNotImplementedException(string.Format("Incorrect type argument; the type {0} is not registered for use in this controller version.", typeof(T).Name));

            rrBase.Init(Renderer);
        }


        public virtual IAvatarController GetAvatarController(int avatarId)
        {
            return null;
        }

        #endregion
    }
}
