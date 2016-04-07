using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace VRage.Library.Collections
{
    public class AsyncBuffer<T> : IDisposable
    {
        private readonly Queue<T> m_queue;
        private readonly Queue<TaskCompletionSource<T>> m_waitingTasks;


        public bool Disposed { get; private set; }


        public AsyncBuffer()
        {
            m_queue = new Queue<T>();
            m_waitingTasks = new Queue<TaskCompletionSource<T>>();
        }

        public void Dispose()
        {
            Cancel();
        }

        public void Cancel()
        {
            lock (m_queue)
            {
                foreach (var taskCompletionSource in m_waitingTasks)
                    taskCompletionSource.SetCanceled();

                m_waitingTasks.Clear();
                m_queue.Clear();
                Disposed = true;
            }
        }


        public void Add(T item)
        {
            TaskCompletionSource<T> tcs = null;

            lock (m_queue)
            {
                if (m_waitingTasks.Count > 0)
                {
                    tcs = m_waitingTasks.Dequeue();
                }
                else
                {
                    m_queue.Enqueue(item);
                }
            }

            if (tcs != null)
            {
                tcs.TrySetResult(item);
            }
        }

        public Task<T> Get()
        {
            lock (m_queue)
            {
                if (m_queue.Count > 0)
                {
                    return Task.FromResult(m_queue.Dequeue());
                }

                var tcs = new TaskCompletionSource<T>();
                m_waitingTasks.Enqueue(tcs);
                return tcs.Task;
            }
        }
    }
}
