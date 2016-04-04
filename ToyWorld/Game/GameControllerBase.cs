using System.Collections.Generic;
using GoodAI.ToyWorld.Control;
using Render.Renderer;
using Render.RenderRequests;
using World.GameActors.GameObjects;
using World.ToyWorldCore;
using World.WorldInterfaces;

namespace Game
{
    // TODO: why there is abstract class instead of BasicGameController?
    public abstract class GameControllerBase : IGameController
    {
        public IRenderer Renderer { get; private set; }
        protected IWorld World { get; private set; }
        private GameSetup GameSetup { get; set; }
        private Dictionary<int, Avatar> m_avatars;
        private Dictionary<int, AvatarController> m_avatarControllers;


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
            GameSetup = setup;
            World = new ToyWorld(GameSetup.SaveFile, GameSetup.TilesetFile);

            m_avatars = new Dictionary<int, Avatar>();
            foreach (var avatarId in World.GetAvatarsIds())
            {
                m_avatars.Add(avatarId, World.GetAvatar(avatarId));
            }

            m_avatarControllers = new Dictionary<int, AvatarController>();
            foreach (var avatar in m_avatars)
            {
                m_avatarControllers.Add(avatar.Key, new AvatarController(avatar.Value));
            }

            Renderer.Init();
            Renderer.CreateWindow("TestGameWindow", 1024, 1024);
            Renderer.CreateContext();
        }

        public virtual void Reset()
        {
            // TODO: Semantics of Reset? What should it do?
            // current implementation: loads world from file again
            World = new ToyWorld(GameSetup.SaveFile, GameSetup.TilesetFile);
        }


        public virtual void MakeStep()
        {
            // Assume Init has been called, we don't want to check for consistency every step
            World.Update();

            foreach (var avatarController in m_avatarControllers)
            {
                avatarController.Value.ResetControls();
            }

            Renderer.ProcessRequests();
        }


        public virtual T RegisterRenderRequest<T>()
            where T : class, IRenderRequest
        {
            var rr = RenderRequestFactory.CreateRenderRequest<T>();
            InitRr(rr);
            Renderer.EnqueueRequest(rr);

            return rr;
        }

        public virtual T RegisterAvatarRenderRequest<T>(int avatarId)
            where T : class, IAvatarRenderRequest
        {
            // TODO: check agentID or make the param an AgentController?

            var rr = RenderRequestFactory.CreateAvatarRenderRequest<T>(avatarId);
            InitRr(rr);
            Renderer.EnqueueRequest(rr);

            return rr;
        }

        void InitRr<T>(T rr)
            where T : class
        {
            var rrBase = rr as RenderRequest; // Assume that all renderRequests created by factory inherit from RenderRequest

            if (rrBase == null)
                throw new RenderRequestNotImplementedException(string.Format("Incorrect type argument; the type {0} is not registered for use in this controller version.", typeof(T).Name));

            rrBase.Init(Renderer);
        }


        public IAvatarController GetAvatarController(int avatarId)
        {
            return m_avatarControllers[avatarId];
        }

        #endregion
    }
}
