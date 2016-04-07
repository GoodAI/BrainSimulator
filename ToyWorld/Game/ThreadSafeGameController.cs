using System;
using System.Collections.Generic;
using System.Diagnostics;
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
        private readonly CancellationTokenSource m_cancellationToken = new CancellationTokenSource();
        private readonly Task m_requestCollectionTask;

        public ThreadSafeGameController(RendererBase renderer, GameSetup gameSetup)
            : base(renderer, gameSetup)
        {
            m_requestCollectionTask = Task.Run(() => RunRequestCollectionAsync(), m_cancellationToken.Token);
        }

        public override void Dispose()
        {
            m_cancellationToken.Cancel(true);
            m_buffer.Dispose();
        }


        #region Long-running method

        void RunRequestCollectionAsync()
        {
            while (!m_buffer.Disposed)
            {
                TupleType result = null;

                var task = m_buffer.Get();

                try
                {
                    result = task.Result; // blocks while waiting for the result

                    if (task.IsCanceled)
                        ;

                    var res = result.Item1();
                    result.Item2.SetResult(res);
                }
                catch (Exception e)
                {
                    if (result != null)
                        result.Item2.SetException(e);

                    Debug.Fail("Shenanigans");
                    // TODO: log
                }
            }
        }

        #endregion

        #region Delegation

        void DelegateStuff(Action action)
        {
            Func<object> dummy = () =>
            {
                action();
                return null;
            };

            DelegateStuffInternal(dummy);
        }

        T DelegateStuff<T>(Func<T> func)
            where T : class
        {
            return DelegateStuffInternal(func) as T;
        }

        object DelegateStuffInternal(Func<object> func)
        {
            var t = new TaskCompletionSource<object>();
            m_buffer.Add(new TupleType(func, t));
            t.Task.Wait(m_cancellationToken.Token);
            return t.Task.Result;
        }

        #endregion

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
            return DelegateStuff<T>(base.RegisterRenderRequest<T>);
        }

        public override IAvatarController GetAvatarController(int avatarId)
        {
            // TODO: get threadsafe avatar controller
            return DelegateStuff(() => base.GetAvatarController(avatarId));
        }

        #endregion
    }
}
