using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Platform.Core.Utils;
using Microsoft.CSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using YAXLib;

namespace GoodAI.Modules.Scripting
{
    public class MyCSharpNodeGroup : MyNodeGroup, IScriptableNode, IMyCustomExecutionPlanner
    {
        internal enum ScriptMethods
        {
            Init,
            Execute,
            __InitGeneratedVariables
        }

        public MyCSharpNodeGroup()
        {
            InputBranches = 1;
            Script = EXAMPLE_CODE;
            ScriptEngine = new CSharpEngine<ScriptMethods>(this);
        }

        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();

            if (GenerateVariables)
            {
                CollectNodeAndTaskIdentifiers();
            }
            else
            {
                m_nameExpressions = String.Empty;
            }
        }

        public override string Description
        {
            get
            {
                return "CSharpGroup";
            }
        }

        [MyBrowsable, Category("Script Generator"), Description("Generates variables for nodes and tasks.")]
        [YAXSerializableField(DefaultValue = true)]
        public bool GenerateVariables { get; set; }

        #region task planner

        private MyExecutionBlock m_defaultPlan;

        private void InitAllTasksInDefaultPlan()
        {
            if (m_defaultPlan != null)
            {
                m_defaultPlan.Iterate(true, executable =>
                {
                    MyTask task = executable as MyTask;
                    if (task != null && task != ExecuteScript)
                    {
                        task.Init(GPU);
                    }
                });
            }
        }

        public void ExecuteAllTasksInDefaultPlan()
        {
            if (m_defaultPlan != null)
            {
                m_defaultPlan.Iterate(true, executable => { if (executable != ExecuteScript) executable.Execute(); });
            }
        }

        public MyExecutionBlock CreateCustomExecutionPlan(MyExecutionBlock defaultPlan)
        {
            m_defaultPlan = defaultPlan;
            return new MyExecutionBlock(ExecuteScript);
        }

        public MyExecutionBlock CreateCustomInitPhasePlan(MyExecutionBlock defaultInitPhasePlan)
        {
            return defaultInitPhasePlan;
        }

        public override void Cleanup()
        {
            m_defaultPlan = null;
        }

        #endregion

        #region Scripting stuff
        [YAXSerializableField]
        protected string m_script;

        public event EventHandler<MyPropertyChangedEventArgs<string>> ScriptChanged;

        public string Script
        {
            get { return m_script; }
            set
            {
                string oldValue = m_script;
                m_script = value;
                if (ScriptChanged != null)
                {
                    ScriptChanged(this, new MyPropertyChangedEventArgs<string>(oldValue, m_script));
                }
            }
        }

        public string NameExpressions
        {
            get { return m_nameExpressions; }
        }

        public string Keywords
        {
            get { return ScriptEngine.DefaultKeywords; }
        }

        public string Language
        {
            get { return ScriptEngine.Language; }
        }

        internal CSharpEngine<ScriptMethods> ScriptEngine { get; set; }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            string finalScript = AppendGeneratedCode(Script, validator);
            ScriptEngine.Compile(finalScript, validator);

            if (m_codeGenerationLog != null && m_codeGenerationLog.Count > 1)
            {
                validator.AddWarning(this, string.Join("\n", m_codeGenerationLog));
            }
        }

        private Dictionary<string, int> m_nodeVariables;
        private Dictionary<string, Tuple<int, string>> m_taskVariables;

        private string m_nameExpressions = String.Empty;
        private List<string> m_codeGenerationLog;

        private void CollectNodeAndTaskIdentifiers()
        {
            CSharpCodeProvider codeProvider = new CSharpCodeProvider();            
            
            m_nodeVariables = new Dictionary<string, int>();
            m_taskVariables = new Dictionary<string, Tuple<int, string>>();

            m_codeGenerationLog = new List<string>();
            m_codeGenerationLog.Add("Script generator:");

            Iterate(true, false, node => 
            {
                string nodeIdentifier = node.Name;
                nodeIdentifier = nodeIdentifier.Replace(' ', '_');

                if (!codeProvider.IsValidIdentifier(nodeIdentifier) || m_nodeVariables.ContainsKey(nodeIdentifier))
                {
                    string newIdentifier = "Node_" + node.Id;
                    m_codeGenerationLog.Add("\"" + nodeIdentifier + "\" is invalid name for identifier (or it exists already). \"" + newIdentifier + "\" generated instead.");
                    nodeIdentifier = newIdentifier;
                }
                m_nodeVariables[nodeIdentifier] = node.Id;

                MyWorkingNode wNode = node as MyWorkingNode;

                if (wNode != null)
                {
                    foreach(string taskName in node.GetInfo().KnownTasks.Keys) 
                    {
                        string taskIdentifier = nodeIdentifier + "_" + taskName;

                        if (!codeProvider.IsValidIdentifier(taskIdentifier) || m_nodeVariables.ContainsKey(taskIdentifier))
                        {
                            m_codeGenerationLog.Add("\"" + nodeIdentifier + "\" is invalid name for identifier (or it exists already). No task variable generated.");
                        }
                        else
                        {
                            m_taskVariables[taskIdentifier] = new Tuple<int, string>(wNode.Id, taskName);
                        }
                    }                    
                }
            });

            m_nameExpressions = String.Empty;

            if (m_nodeVariables.Any())
            {
                m_nameExpressions = m_nodeVariables.Keys.Aggregate((a, b) => a + " " + b);
            }

            if (m_taskVariables.Any())
            {
                m_nameExpressions += " " + m_taskVariables.Keys.Aggregate((a, b) => a + " " + b);
            }
        }

        private string GenerateVariablesCode()
        {
            StringBuilder defs = new StringBuilder();
            StringBuilder init = new StringBuilder();

            init.AppendLine("public static void __InitGeneratedVariables(MyCSharpNodeGroup owner) { ");

            if (GenerateVariables)
            {
                foreach (string nodeIdentifier in m_nodeVariables.Keys)
                {
                    MyNode node = GetChildNodeById(m_nodeVariables[nodeIdentifier]);
                    string nodeTypeName = node.GetType().FullName.Replace('+', '.');

                    if (node != null)
                    {
                        defs.AppendLine("static " + nodeTypeName + " " + nodeIdentifier + ";");
                        init.AppendLine(nodeIdentifier + " = (" + nodeTypeName + ")owner.GetChildNodeById(" + node.Id + ");");
                    }
                }

                foreach (string taskIdentifier in m_taskVariables.Keys)
                {
                    MyWorkingNode wNode = (MyWorkingNode)GetChildNodeById(m_taskVariables[taskIdentifier].Item1);

                    MyTask task = wNode.GetTaskByPropertyName(m_taskVariables[taskIdentifier].Item2);

                    if (task != null)
                    {
                        string taskTypeName = task.GetType().FullName.Replace('+', '.');
                        defs.AppendLine("static " + taskTypeName + " " + taskIdentifier + ";");
                        init.AppendLine(taskIdentifier + " = (" + taskTypeName + ")((MyWorkingNode)owner.GetChildNodeById(" + wNode.Id + ")).GetTaskByPropertyName(\"" + m_taskVariables[taskIdentifier].Item2 + "\");");
                    }
                }
            }

            init.AppendLine("}");

            return defs.ToString() + init.ToString();
        }

        private const string GENERATED_CODE_LABEL = "//[generated_code_location]";

        private string AppendGeneratedCode(string script, MyValidator validator)
        {            
            string completeScript = script;

            int location = script.IndexOf(GENERATED_CODE_LABEL);

            if (location >= 0)
            {
                completeScript = script.Substring(0, location)
                    + GenerateVariablesCode()
                    + script.Substring(location);
            }
            else
            {
                validator.AddError(this, "Script generator: label \"" + GENERATED_CODE_LABEL + "\" not found. No generated code available.");
            }            

            return completeScript;
        }

        #endregion

        #region Tasks

        public MyInitScriptTask InitScript { get; private set; }

        /// <summary>
        /// Runs Init() method of the script one time
        /// </summary>
        [Description("Init script"), MyTaskInfo(OneShot = true)]
        public class MyInitScriptTask : MyTask<MyCSharpNodeGroup>
        {
            public override void Init(int nGPU)
            {

            }

            public override void Execute()
            {
                if (Owner.ScriptEngine.HasMethod(ScriptMethods.Init))
                {
                    try
                    {
                        Owner.ScriptEngine.Run(ScriptMethods.Init, Owner);
                    }
                    catch (Exception e)
                    {
                        MyLog.WARNING.WriteLine("Script Init() call failed: " + e.GetType().Name + ": " + e.Message);
                    }
                }
                else
                {
                    MyLog.WARNING.WriteLine(Owner.Name + ": No Init() method available");
                }
            }
        }

        public MyExecuteScriptTask ExecuteScript { get; private set; }

        /// <summary>
        /// Runs Execute() method of the script
        /// </summary>
        [Description("Execute script")]
        public class MyExecuteScriptTask : MyTask<MyCSharpNodeGroup>
        {
            public override void Init(int nGPU)
            {
                Owner.InitAllTasksInDefaultPlan();

                if (Owner.ScriptEngine.HasMethod(ScriptMethods.__InitGeneratedVariables))
                {
                    try
                    {
                        Owner.ScriptEngine.Run(ScriptMethods.__InitGeneratedVariables, Owner);
                    }
                    catch (Exception e)
                    {
                        MyLog.WARNING.WriteLine("__InitGeneratedVariables() call failed: " + e.GetType().Name + ": " + e.Message);
                    }
                }
                else
                {
                    MyLog.WARNING.WriteLine(Owner.Name + ": No __InitGeneratedVariables() method available");
                }
            }

            public override void Execute()
            {
                if (Owner.ScriptEngine.HasMethod(ScriptMethods.Execute))
                {
                    try
                    {
                        Owner.ScriptEngine.Run(ScriptMethods.Execute, Owner);
                    }
                    catch (Exception e)
                    {
                        MyLog.WARNING.WriteLine("Script Execute() call failed: " + e.GetType().Name + ": " + e.Message);
                    }
                }
                else
                {
                    MyLog.WARNING.WriteLine(Owner.Name + ": No Execute() method available");
                }
            }
        }

        #endregion

        #region ExampleCode
        private const string EXAMPLE_CODE = @"using System;
using GoodAI.Core.Utils;
using GoodAI.Core.Nodes;
using GoodAI.Modules.Scripting;

namespace Runtime
{
    public class Script
    {
        public static void Init(MyCSharpNodeGroup owner)
        {
            
        }
        
        public static void Execute(MyCSharpNodeGroup owner)
        {
            if (owner.ExecuteScript.SimulationStep % 2 == 0) 
            {
                owner.ExecuteAllTasksInDefaultPlan();
            }
        }
        
        " + GENERATED_CODE_LABEL + @" DO NOT REMOVE THIS COMMENT!
    }
}";
        #endregion
    }
}
