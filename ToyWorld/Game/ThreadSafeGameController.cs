using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Control;
using Render.Renderer;
using Render.RenderRequests;
using VRage.Library.Collections;

namespace Game
{
    public class ThreadSafeGameController : GameControllerBase
    {
        private readonly AsyncBuffer<Tuple<Action, TaskCompletionSource<bool>>> m_buffer = new AsyncBuffer<Tuple<Action, TaskCompletionSource<bool>>>();
        readonly CancellationTokenSource m_cancellationToken = new CancellationTokenSource();


        public ThreadSafeGameController(RendererBase renderer, GameSetup gameSetup)
            : base(renderer, gameSetup)
        {
            RunRequestCollectionAsync();
        }

        public override void Dispose()
        {
            m_cancellationToken.Cancel();
            m_buffer.Dispose();
        }


        async Task RunRequestCollectionAsync()
        {
            while (!m_buffer.Disposed)
            {
                var action = await m_buffer.Get().ConfigureAwait(true);
                action.Item1();
                action.Item2.SetResult(true);
            }
        }

        void DoAsyncStuffSync(Action action)
        {
            var t = new TaskCompletionSource<bool>();
            m_buffer.Add(new Tuple<Action, TaskCompletionSource<bool>>(action, t));
            t.Task.Wait(m_cancellationToken.Token);
        }


        #region GameControllerBase overrides

        public override void Init()
        {
            DoAsyncStuffSync(base.Init);
        }

        public override void Reset()
        {
        }

        public override void MakeStep()
        {
        }

        public override T RegisterRenderRequest<T>(int avatarId)
        {
            return base.RegisterRenderRequest<T>(avatarId);
        }

        public override T RegisterRenderRequest<T>()
        {
            return base.RegisterRenderRequest<T>();
        }

        #endregion
    }
}
