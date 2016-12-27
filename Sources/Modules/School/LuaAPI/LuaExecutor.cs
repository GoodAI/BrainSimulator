using System;
using System.Text;
using System.Threading;
using World.Atlas;
using World.Lua;

namespace LuaAPI
{
    public delegate void LuaExecutorInitEventHandler();

    public class LuaExecutor
    {
        private static NLua.Lua m_state;
        private readonly AutoResetEvent m_scriptSynchronization;

        public LuaExecutor(IAtlas atlas, AutoResetEvent scriptSynchronization)
        {
            m_scriptSynchronization = scriptSynchronization;
            m_state = new NLua.Lua();
            m_state.LoadCLRPackage();
            m_state.DoString(@"import ('World','World.ToyWorldCore')");
            m_state.DoString(@"import ('World','World.Lua')");
            AvatarHelper avatarHelper = new AvatarHelper(m_state, atlas, scriptSynchronization);
            m_state["ah"] = avatarHelper;
        }

        private Thread m_thread;

        public void ExecuteChunk(string command, Action<string> performAfterFinished = null)
        {
            m_thread = new Thread(() => RunScript(command, performAfterFinished));
            m_thread.Start();
            Thread.Sleep(1);
        }

        private void RunScript(string command, Action<string> performAfterFinished = null)
        {
            StringBuilder result = new StringBuilder();
            try
            {
                m_scriptSynchronization.WaitOne();
                object[] objects = m_state.DoString(command);

                if (objects != null)
                {
                    result.Append("{ ");
                    foreach (object o in objects)
                    {
                        if (o == null) continue;
                        result.Append(o).Append(", ");
                    }
                    result.TrimEnd(result.Length > 2 ? 2 : 1);
                    result.Append(" }");
                }
            }
            catch (NLua.Exceptions.LuaScriptException e)
            {
                result.Append(e);
            }
            performAfterFinished?.Invoke(result.ToString());
        }

    }
}
