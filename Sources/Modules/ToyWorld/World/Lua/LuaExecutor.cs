using System;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using World.Atlas;

namespace World.Lua
{
    public delegate void LuaExecutorInitEventHandler();

    public class LuaExecutor
    {
        public NLua.Lua State;
        private readonly AutoResetEvent m_scriptSynchronization;

        public LuaExecutor(IAtlas atlas, AutoResetEvent scriptSynchronization)
        {
            m_scriptSynchronization = scriptSynchronization;
            State = new NLua.Lua();
            State.LoadCLRPackage();
            State.DoString(@"import ('World','World.ToyWorldCore')");
            State.DoString(@"import ('World','World.Lua')");

            State["le"] = this;
            State["atlas"] = atlas;

            State.RegisterFunction("Help", typeof(LuaExecutor).GetMethod("Help"));

            if (atlas.Avatars.Count > 0)
            {
                AvatarCommander avatarCommander = new AvatarCommander(this, atlas);
                State["ac"] = avatarCommander;
            }

            AtlasManipulator atlasManipulator = new AtlasManipulator(this, atlas);
            State["am"] = atlasManipulator;
        }

        private Thread m_thread;
        private bool m_stopScript;

        public Thread ExecuteChunk(string command, Action<string> performAfterFinished = null)
        {
            m_thread = new Thread(() => RunScript(command, performAfterFinished));
            m_thread.Start();
            Thread.Sleep(1);
            return m_thread;
        }

        private void RunScript(string command, Action<string> performAfterFinished = null)
        {
            StringBuilder result = new StringBuilder();

            m_scriptSynchronization.WaitOne();

            object[] objects = null;
            try
            {
                objects = State.DoString(command);
            }
            catch (NLua.Exceptions.LuaScriptException e)
            {
                try
                {
                    objects = State.DoString("return " + command);

                }
                catch (NLua.Exceptions.LuaScriptException)
                {
                    result.Append(e);
                }
            }

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

            if (result.Length == 0)
            {
                result.Append("Done");
            }

            performAfterFinished?.Invoke(result.ToString());
        }

        public void Do(Func<object[], bool> stepFunc, params object[] parameters)
        {
            for (int i = 0; i < 100000; i++)
            {
                m_scriptSynchronization.WaitOne();
                if (m_stopScript)
                {
                    m_stopScript = false;
                    return;
                }
                object o = stepFunc(parameters);
                bool end = (bool)o;
                if (end)
                {
                    return;
                }


            }
            throw new Exception("Too long time in Do function.");
        }

        public void Repeat(Action<object[]> stepFunc, int repetitions, params object[] parameters)
        {
            for (int i = 0; i < repetitions; i++)
            {
                if (m_stopScript)
                {
                    m_stopScript = false;
                    return;
                }
                m_scriptSynchronization.WaitOne();
                stepFunc(parameters);
            }
        }

        public void Perform(Action<object[]> stepFunc, params object[] parameters)
        {
            m_scriptSynchronization.WaitOne();
            stepFunc(parameters);
        }

        public void StopScript()
        {
            m_stopScript = true;
        }

        public static string Help(object o)
        {
            if (o == null) return "null";
            Type type = o.GetType();
            PropertyInfo[] propertyInfos = type.GetProperties();
            string propertiesJoined = string.Join(",\n\t\t", propertyInfos.Select(x => x.ToString()));
            propertiesJoined = Regex.Replace(propertiesJoined, @"\w+\.", "");
            MethodInfo[] methodInfos = type.GetMethods();
            string methodsJoined = string.Join(",\n\t\t", methodInfos.Select(x => x.ToString()));
            methodsJoined = Regex.Replace(methodsJoined, @"\w+\.", "");
            return "Name: \"" + type.Name + "\"\n\tProperties: { " + propertiesJoined + " } " + "\n\tMethods: { " +
                   methodsJoined + " }.";
        }
    }
}
