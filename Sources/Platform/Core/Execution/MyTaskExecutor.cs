using System;
using System.Diagnostics;
using System.Threading;

namespace GoodAI.Core.Execution
{
    /// Run the tasks
    public class MyThreadPool
    {
        private int m_numOfThreads;
       
        protected Thread[] m_threads;
        protected ThreadExecInfo[] m_threadExecInfos;
        protected AutoResetEvent[] m_eventsDone;

        private bool m_disposed = false;

        public delegate void ThreadHandler(int threadId);

        private ThreadHandler m_initThread;
        private ThreadHandler m_executeThread;

        public MyThreadPool(int numOfThreads, ThreadHandler initThread, ThreadHandler executeThread)
        {
            m_numOfThreads = numOfThreads;

            m_initThread = initThread;
            m_executeThread = executeThread;

            m_threads = new Thread[m_numOfThreads];
            m_threadExecInfos = new ThreadExecInfo[m_numOfThreads];
            m_eventsDone = new AutoResetEvent[m_numOfThreads];       
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected void Dispose(bool disposing)
        {
            if (m_disposed) return;

            if (disposing)
            {
                for (Int32 i = 0; i < m_numOfThreads; i++)
                {
                    m_threadExecInfos[i].Dispose();
                }
            }
            m_disposed = true;
        }

        public void StartThreads()
        {
            for (int i = 0; i < m_numOfThreads; i++)
            {
                m_threads[i] = new Thread(Worker);
                m_threads[i].Name = "Task Executor Thread #" + i;
                m_threadExecInfos[i] = new ThreadExecInfo(i);
                m_eventsDone[i] = m_threadExecInfos[i].EventDone;

                m_threads[i].Start(m_threadExecInfos[i]);
            }     
        }

        public void ResumeThreads()
        {
            for (int i = 0; i < m_numOfThreads; i++)
            {
                m_threadExecInfos[i].EventGo.Set();
            }         
            WaitHandle.WaitAll(m_eventsDone);
        }

        public void Finish()
        {
            for (int i = 0; i < m_numOfThreads; i++)
            {
                m_threadExecInfos[i].EventFinish.Set();
            }         
            WaitHandle.WaitAll(m_eventsDone);
        }

        public void FinishFromSTAThread()
        {
            for (int i = 0; i < m_numOfThreads; i++)
            {
                m_threadExecInfos[i].EventFinish.Set();
                m_eventsDone[i].WaitOne();
            }            
        }

        protected static int INDEX_GO = 0;
        protected static int INDEX_FINISH = 1;        

        private void Worker(object param)
        {
            ThreadExecInfo threadExecInfo = param as ThreadExecInfo;
            WaitHandle[] waitHandles = { threadExecInfo.EventGo, threadExecInfo.EventFinish };

            m_initThread(threadExecInfo.Id);

            while (true)
            {
                int index = WaitHandle.WaitAny(waitHandles);
                
                if(index == INDEX_FINISH)
                {
                    threadExecInfo.EventDone.Set();                    
                    return;
                }
                Debug.Assert(index == INDEX_GO, "Unexpected wait result.");

                m_executeThread(threadExecInfo.Id);

                threadExecInfo.EventDone.Set();
            }
        }        
    }

    /// Stores information about thread
    public class ThreadExecInfo : IDisposable
    {        
        private bool m_disposed = false;

        public int Id { get; private set; } // ID of GPU, which will thread process
        
        public AutoResetEvent EventGo { get; private set; } // "start to work" event
        public AutoResetEvent EventDone { get; private set; } // "work is done" event
        public AutoResetEvent EventFinish { get; private set; } // "thread should finish" event

        public ThreadExecInfo(int id)
        {
            Id = id;            
            EventGo = new AutoResetEvent(false);
            EventDone = new AutoResetEvent(false);
            EventFinish = new AutoResetEvent(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (m_disposed)
                return;

            if (disposing)
            {
                EventGo.Dispose();
                EventDone.Dispose();
                EventFinish.Dispose();
            }

            m_disposed = true;
        }
    }
}
