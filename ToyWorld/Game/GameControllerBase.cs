using System;
using System.Linq;
using System.Collections.Generic;
using GoodAI.ToyWorld.Control;
using Render.Renderer;
using Render.RenderRequests;
using TmxMapSerializer.Elements;
using TmxMapSerializer.Serializer;
using World.GameActors.GameObjects;
using World.ToyWorldCore;
using World.WorldInterfaces;

namespace Game
{
    // TODO: why there is abstract class instead of BasicGameController?
    public abstract class GameControllerBase : IGameController
    {
        private bool m_initialized;
        private readonly GameSetup m_gameSetup;

        public RendererBase Renderer { get; private set; }
        public ToyWorld World { get; private set; }

        private Dictionary<int, IAvatar> m_avatars;
        private Dictionary<int, AvatarController> m_avatarControllers;


        protected GameControllerBase(RendererBase renderer, GameSetup setup)
        {
            Renderer = renderer;
            m_gameSetup = setup;
        }

        public virtual void Dispose()
        {
            if (Renderer != null)
                Renderer.Dispose();
            Renderer = null;
        }


        #region IGameController overrides

        public virtual void Init()
        {
            m_initialized = true;

            // Init World
            var serializer = new TmxSerializer();
            Map map = serializer.Deserialize(m_gameSetup.SaveFile);
            m_gameSetup.SaveFile.Close();

            World = new ToyWorld(map, m_gameSetup.TilesetFile);

            m_avatars = new Dictionary<int, IAvatar>();
            foreach (int avatarId in World.GetAvatarsIds())
            {
                m_avatars.Add(avatarId, World.GetAvatar(avatarId));
            }

            m_avatarControllers = new Dictionary<int, AvatarController>();
            foreach (KeyValuePair<int, IAvatar> avatar in m_avatars)
            {
                var avatarController = new AvatarController(avatar.Value);
                m_avatarControllers.Add(avatar.Key, avatarController);
            }

            // Init rendering
            Renderer.Init();
            Renderer.CreateWindow("TestGameWindow", 1024, 1024);
            Renderer.CreateContext();
        }

        public virtual void MakeStep()
        {
            if (!m_initialized)
            {
                Init();
            }

            // Controls should be already set

            World.Update();

            ResetAvatarControllers();

            Renderer.ProcessRequests(World);
        }


        public virtual T RegisterRenderRequest<T>()
            where T : class, IRenderRequest
        {
            Renderer.MakeContextCurrent();
            var rr = RenderRequestFactory.CreateRenderRequest<T>();
            InitRR(rr);
            Renderer.EnqueueRequest(rr);
            Renderer.MakeContextNotCurrent();

            return rr;
        }

        public virtual T RegisterRenderRequest<T>(int avatarId)
            where T : class, IAvatarRenderRequest
        {
            // TODO: check agentID or make the param an AgentController?

            Renderer.MakeContextCurrent();
            T rr = RenderRequestFactory.CreateRenderRequest<T>(avatarId);
            InitRR(rr);
            Renderer.EnqueueRequest(rr);
            Renderer.MakeContextNotCurrent();

            return rr;
        }

        void InitRR<T>(T rr)
            where T : class
        {
            RenderRequest rrBase = rr as RenderRequest; // Assume that all renderRequests created by factory inherit from RenderRequest

            if (rrBase == null)
                throw new RenderRequestNotImplementedException(
                    string.Format("Incorrect type argument; the type {0} is not registered for use in this controller version.",
                    typeof(T).Name));

            rrBase.Init(Renderer, World);
        }


        public virtual IAvatarController GetAvatarController(int avatarId)
        {
            return m_avatarControllers[avatarId];
        }

        private void ResetAvatarControllers()
        {
            foreach (AvatarController avatarController in m_avatarControllers.Values)
            {
                avatarController.ResetControls();
            }
        }

        public int[] GetAvatarIds()
        {
            return m_avatarControllers.Keys.ToArray();
        }

        #endregion
    }
}
