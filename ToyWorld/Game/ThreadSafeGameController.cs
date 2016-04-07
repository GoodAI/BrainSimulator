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

using TupleType = System.Tuple<System.Func<object>, System.Threading.Tasks.TaskCompletionSource<object>>;

namespace Game
{
    public class ThreadSafeGameController : GameControllerBase
    {
        private readonly AsyncBuffer<TupleType> m_buffer = new AsyncBuffer<TupleType>();
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


        #region Long-running method

        async Task RunRequestCollectionAsync()
        {
            await Task.Yield();

            while (!m_buffer.Disposed)
            {
                var action = await m_buffer.Get().ConfigureAwait(true);
                action.Item2.SetResult(action.Item1());
            }
        }

        #endregion

        void DelegateStuff(Action action)
        {
            DelegateStuff<object>(
                () =>
                {
                    action();
                    return null;
                });
        }

        T DelegateStuff<T>(Func<T> action)
            where T : class
        {
            var t = new TaskCompletionSource<object>();
            m_buffer.Add(new TupleType(action, t));
            t.Task.Wait(m_cancellationToken.Token);
            return t.Task.Result as T;
        }


        #region GameControllerBase overrides -- public threadsafe methods

        public override void Init()
        {
            DelegateStuff(base.Init);
        }

        public override void Reset()
        {
            DelegateStuff(base.Reset);
        }

        public override void MakeStep()
        {
            DelegateStuff(base.MakeStep);
        }

        public override T RegisterRenderRequest<T>(int avatarId)
        {
            return DelegateStuff(() => base.RegisterRenderRequest<T>(avatarId));
        }

        public override T RegisterRenderRequest<T>()
        {
            return DelegateStuff(() => base.RegisterRenderRequest<T>());
        }

        public override IAvatarController GetAvatarController(int avatarId)
        {
            // TODO: get threadsafe avatar controller
            return DelegateStuff(() => base.GetAvatarController(avatarId));
        }

        #endregion
    }
}
