using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Control;
using Render.Renderer;
using VRage.Library.Collections;

using TupleType = System.Tuple<System.Func<object>, System.Threading.Tasks.TaskCompletionSource<object>>;

namespace Game
{
    public class ThreadSafeGameController : GameControllerBase
    {
        private readonly AsyncBuffer<TupleType> m_buffer = new AsyncBuffer<TupleType>();
        private readonly CancellationTokenSource m_cancellationToken = new CancellationTokenSource();
        private Task m_requestCollectionTask;


        public ThreadSafeGameController(RendererBase renderer, GameSetup gameSetup)
            : base(renderer, gameSetup)
        {
            m_requestCollectionTask = Task.Factory.StartNew(RunRequestCollectionAsync, TaskCreationOptions.LongRunning);
        }

        public override void Dispose()
        {
            // Signal disposal of this object, prevents repeated disposal
            var task = Interlocked.Exchange(ref m_requestCollectionTask, null);

            if (task == null)
                return;

            // Call the internal one to skip m_requestCollectionTask checking
            DelegateStuffInternal(() =>
            {
                base.Dispose();
                return null;
            });

            if (!m_cancellationToken.IsCancellationRequested)
                m_cancellationToken.Cancel(true);
        }


        #region Long-running method

        private void RunRequestCollectionAsync()
        {
            while (!m_cancellationToken.IsCancellationRequested)
            {
                Task<TupleType> task;

                // Get a new task to wait for
                try
                {
                    task = m_buffer.Get();
                }
                catch (ObjectDisposedException)
                {
                    Debug.Assert(m_cancellationToken.IsCancellationRequested);
                    return;
                }
                catch (Exception)
                {
                    // Should not ever happen
                    Debug.Assert(false);
                    m_buffer.Clear();
                    continue;
                }

                // Wait for the producers to queue up an item
                try
                {
                    task.Wait(m_cancellationToken.Token);
                }
                catch (Exception)
                {
                    if (m_cancellationToken.IsCancellationRequested)
                        return;

                    continue;
                }

                // Execute the item and return its result
                TupleType result = null;

                try
                {
                    result = task.Result;

                    var res = result.Item1();
                    result.Item2.SetResult(res);
                }
                catch (Exception e)
                {
                    Debug.Assert(result != null);
                    result.Item2.SetException(e);
                }
            }
        }

        #endregion

        #region Delegation

        private void DelegateStuff(Action action)
        {
            // Must be here (not in DelegateInternal) to enable delegating disposal
            if (m_requestCollectionTask == null)
                throw new ObjectDisposedException("ThreadSafeController");

            Func<object> dummy = () =>
            {
                action();
                return null;
            };

            DelegateStuffInternal(dummy);
        }

        private T DelegateStuff<T>(Func<T> func)
            where T : class
        {
            // Must be here (not in DelegateInternal) to enable delegating disposal
            if (m_requestCollectionTask == null)
                throw new ObjectDisposedException("ThreadSafeController");

            return DelegateStuffInternal(func) as T;
        }

        private object DelegateStuffInternal(Func<object> func)
        {
            Debug.Assert(func != null);

            var t = new TaskCompletionSource<object>();
            m_buffer.Add(new TupleType(func, t)); // Throws ObjectDisposedException when called after Dispose has been called

            try
            {
                t.Task.Wait(m_cancellationToken.Token);
            }
            catch (OperationCanceledException e)
            {
                throw new ObjectDisposedException("Trying to call actions on a disposed object.", e);
            }
            catch (AggregateException e)
            {
                foreach (var innerException in e.InnerExceptions)
                {
                    if (innerException is OperationCanceledException)
                        throw new ObjectDisposedException("Trying to call actions on a disposed object.", e);

                    throw innerException;
                }
            }

            return t.Task.Result;
        }

        #endregion

        #region GameControllerBase overrides -- public threadsafe methods

        public override void Init()
        {
            DelegateStuff(base.Init);
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
