using System;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using NLua.Event;
using World.Atlas;

namespace World.Lua
{
    public delegate void LuaExecutorInitEventHandler();

    public class LuaExecutor
    {
        public NLua.Lua State;
        private Thread m_thread;
        private readonly IAtlas m_atlas;
        private readonly LuaConsole m_luaConsole;
        private const int MAXIMUM_NUMBER_OF_DO_METHED_CALLS = 100000;

        // Do not touch those events if you don't know what you're doing! Otherwise useful for lua scripts.
        // Intended use case:
        // while not le.ShouldStopScript do
        //    le.DoWorkSync:WaitOne()
        //    stuff()
        //    le.WorkDoneSync:Set()
        // end
        public AutoResetEvent DoWorkSync { get; }
        public AutoResetEvent WorkDoneSync { get; }
        public bool ShouldStopScript { get; set; }

        public LuaExecutor(IAtlas atlas, LuaConsole luaConsole = null)
        {
            m_atlas = atlas;
            DoWorkSync = new AutoResetEvent(false);
            WorkDoneSync = new AutoResetEvent(false);
            m_luaConsole = luaConsole;
            SetInitialState();
        }

        private void SetInitialState()
        {
            State = new NLua.Lua();
            State.LoadCLRPackage();
            State.DoString(@"import ('World','World.ToyWorldCore')");
            State.DoString(@"import ('World','World.Lua')");

            State["le"] = this;
            State["atlas"] = m_atlas;

            State.RegisterFunction("Help", typeof(LuaExecutor).GetMethod("Help"));

            if (m_atlas.Avatars.Count > 0)
            {
                AvatarCommander avatarCommander = new AvatarCommander(this, m_atlas);
                State["ac"] = avatarCommander;
            }

            AtlasManipulator atlasManipulator = new AtlasManipulator(m_atlas);
            State["am"] = atlasManipulator;

            State["lc"] = m_luaConsole;

            State.DebugHook += OnDebugHook;

            State.SetDebugHook(EventMasks.LUA_MASKLINE, 1000);
        }

        private void OnDebugHook(object sender, DebugHookEventArgs e)
        {
            if (!ShouldStopScript) return;
            State.DoString(@"function TALuaScriptInternalStopHook(why)  error ('" +
                "User interruption." + "'); end; debug.sethook (TALuaScriptInternalStopHook, '', 1);");
            State.DoString("lc:Print(\"Core reset!\")");
            ShouldStopScript = false;
            SetInitialState();
        }


        public Thread ExecuteChunk(string command, Action<string> performAfterFinished = null)
        {
            DoWorkSync.Reset();
            WorkDoneSync.Reset();
            m_thread = new Thread(() => { RunScript(command, performAfterFinished); WorkDoneSync.Set(); })
            {
                IsBackground = true
            };
            m_thread.Start();
            Thread.Sleep(1);
            return m_thread;
        }

        private void RunScript(string command, Action<string> performAfterFinished = null)
        {
            StringBuilder result = new StringBuilder();

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

            // uncomment to get confirmation when command run successfully
            /*if (result.Length == 0)
            {
                result.Append("Done");
            }*/

            performAfterFinished?.Invoke(result.ToString());
        }

        /// <summary>
        /// This method synchronize execution of given function and ToyWorld.
        /// The function is called once per ToyWorld execution step until the function
        /// returns true.
        /// </summary>
        /// <param name="stepFunc">A function changing or observing ToyWorld. Returns bool.</param>
        /// <param name="parameters">Any parameters the function accepts.</param>
        public void Do(Func<object[], bool> stepFunc, params object[] parameters)
        {
            for (int i = 0; i < MAXIMUM_NUMBER_OF_DO_METHED_CALLS; i++)
            {
                DoWorkSync.WaitOne();
                object o = stepFunc(parameters);
                WorkDoneSync.Set();

                if (ShouldStopScript)
                {
                    ShouldStopScript = false;
                    return;
                }
                bool end = (bool)o;
                if (end)
                {
                    return;
                }
            }
            throw new Exception("Too long time in Do function.");
        }

        public void NotifyAndWait()
        {
            Notify();

            if (m_thread?.IsAlive == true)
            {
                WorkDoneSync.WaitOne();
            }
        }

        public void Notify()
        {
            DoWorkSync.Set();
        }

        public void Repeat(Action<object[]> stepFunc, int repetitions, params object[] parameters)
        {
            for (int i = 0; i < repetitions; i++)
            {
                DoWorkSync.WaitOne();
                stepFunc(parameters);
                WorkDoneSync.Set();

                if (ShouldStopScript)
                {
                    ShouldStopScript = false;
                    return;
                }
            }
        }

        public void Repeat(Action stepFunc, int repetitions)
        {
            for (int i = 0; i < repetitions; i++)
            {
                DoWorkSync.WaitOne();
                stepFunc();
                WorkDoneSync.Set();
                if (ShouldStopScript)
                {
                    ShouldStopScript = false;
                    return;
                }
            }
        }

        public void Perform(Action<object[]> stepFunc, params object[] parameters)
        {
            DoWorkSync.WaitOne();
            stepFunc(parameters);
            WorkDoneSync.Set();
        }

        /// <summary>
        /// This function returns info about object containing list of all object's properties and methods.
        /// </summary>
        /// <param name="o">Any C# object in assembly.</param>
        /// <returns></returns>
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
