using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using Microsoft.CSharp;
using System;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Modules.Scripting
{
    /// <author>GoodAI</author>
    /// <status>Working</status>
    /// <summary>Experimental C# scripting node</summary>
    /// <description>Node allows user to write C# code directly and run it during simulation. Example code:
    /// <pre>
    /// using System;
    /// using GoodAI.Core.Utils;
    /// using GoodAI.Core.Nodes;
    /// using GoodAI.Modules.Scripting;
    /// 
    /// namespace Runtime
    /// {
    ///   public class Script
    ///     {
    ///         public static void Init(MyCSharpNode owner)
    ///         {
    ///             MyLog.DEBUG.WriteLine("Init called");
    ///         }
    /// 
    ///         public static void Execute(MyCSharpNode owner)
    ///         {
    ///             MyLog.DEBUG.WriteLine("Execute called");
    /// 
    ///             float[] input = owner.GetInput(0).Host;
    ///             float[] output = owner.GetOutput(0).Host;
    /// 
    ///             output[0] = 3 * (float)Math.Cos(input[0]);
    ///         }
    ///     }
    /// }
    /// </pre>
    /// </description>    
    public class MyCSharpNode : MyScriptableNode, IMyVariableBranchViewNodeBase
    {
        public MyCSharpNode()
        {
            InputBranches = 1;
            Script = EXAMPLE_CODE;
            m_scriptEngine = new MyCSharpEngine<MyDefaultMethods>(this);
        }

        public override string NameExpressions
        {
            get { return "GetInput GetOutput"; }
        }

        public override string Keywords
        {           
            get { return 
                "abstract as base break case catch checked continue default delegate do else event explicit extern false finally fixed for foreach goto if implicit in interface internal is lock namespace new null object operator out override params private protected public readonly ref return sealed sizeof stackalloc switch this throw true try typeof unchecked unsafe using virtual while"
                + " bool byte char class const decimal double enum float int long sbyte short static string struct uint ulong ushort void"; }
        }

        public override string Language
        {
            get { return "CSharp"; }
        }

        public override void UpdateMemoryBlocks()
        {
            UpdateOutputBlocks();
        }        

        #region inputs & outputs
        [ReadOnly(false)]
        [YAXSerializableField, YAXElementFor("IO")]
        public override sealed int InputBranches
        {
            get { return base.InputBranches; }
            set
            {
                base.InputBranches = Math.Max(value, 1);
            }
        }        
        
        private string m_branches;
        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = "1,1"), YAXElementFor("IO")]
        public string OutputBranchesSpec
        {
            get { return m_branches; }
            set
            {
                m_branches = value;
                InitOutputs();
            }
        }

        public void InitOutputs()
        {
            int[] branchConf = GetOutputBranchSpec();

            if (branchConf != null)
            {
                if (branchConf.Length != OutputBranches)
                {
                    //clean-up
                    for (int i = 0; i < OutputBranches; i++)
                    {
                        MyMemoryBlock<float> mb = GetOutput(i);
                        MyMemoryManager.Instance.RemoveBlock(this, mb);
                    }

                    OutputBranches = branchConf.Length;

                    for (int i = 0; i < branchConf.Length; i++)
                    {
                        MyMemoryBlock<float> mb = MyMemoryManager.Instance.CreateMemoryBlock<float>(this);
                        mb.Name = "Output_" + (i + 1);
                        mb.Count = -1;
                        m_outputs[i] = mb;
                    }
                }

                UpdateMemoryBlocks();
            }
        }

        private int[] GetOutputBranchSpec()
        {            
            if (!string.IsNullOrEmpty(OutputBranchesSpec))
            {
                try
                {
                    return OutputBranchesSpec.Split(',').Select(spec => int.Parse(spec, CultureInfo.InvariantCulture)).ToArray();
                }
                catch
                {
                    return null;
                }                
            }          
            else 
            {
                return null;
            }
        }

        private void UpdateOutputBlocks()
        {
            int[] op = GetOutputBranchSpec();

            if (op != null)
            {
                int sum = op.Sum();

                for (int i = 0; i < op.Length; i++)
                {
                    GetOutput(i).Count = op[i];
                }
            }
        }

        #endregion       

        #region Compilation

        internal IScriptingEngine<MyDefaultMethods> m_scriptEngine;

        public override void Validate(MyValidator validator)
        {
            m_scriptEngine.Compile(validator);
        }
        #endregion

        #region Tasks

        public MyInitScriptTask InitScript { get; private set; }

        /// <summary>
        /// Runs Init() method of the script one time
        /// </summary>
        [Description("Init script"), MyTaskInfo(OneShot = true)]
        public class MyInitScriptTask : MyTask<MyCSharpNode>
        {
            public override void Init(int nGPU)
            {
                      
            }
            
            public override void Execute()
            {
                if (Owner.m_scriptEngine.HasMethod(MyDefaultMethods.Init))
                {
                    for (int i = 0; i < Owner.InputBranches; i++)
                    {
                        MyAbstractMemoryBlock mb = Owner.GetAbstractInput(i);

                        if (mb != null)
                        {
                            mb.SafeCopyToHost();
                        }
                    }

                    try
                    {
                        Owner.m_scriptEngine.Run(MyDefaultMethods.Init, Owner);
                    }
                    catch (Exception e)
                    {
                        MyLog.WARNING.WriteLine("Script Init() call failed: " + e.GetType().Name + ": " + e.Message);
                    }

                    for (int i = 0; i < Owner.OutputBranches; i++)
                    {
                        MyAbstractMemoryBlock mb = Owner.GetAbstractOutput(i);

                        if (mb != null)
                        {
                            mb.SafeCopyToDevice();
                        }
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
        public class MyExecuteScriptTask : MyTask<MyCSharpNode>
        {
            public override void Init(int nGPU)
            {

            }

            public override void Execute()
            {
                for (int i = 0; i < Owner.InputBranches; i++)
                {
                    MyAbstractMemoryBlock mb = Owner.GetAbstractInput(i);

                    if (mb != null)
                    {
                        mb.SafeCopyToHost();
                    }
                }

                if (Owner.m_scriptEngine.HasMethod(MyDefaultMethods.Execute))
                {
                    try
                    {
                        Owner.m_scriptEngine.Run(MyDefaultMethods.Execute, Owner);
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

                for (int i = 0; i < Owner.OutputBranches; i++)
                {
                    MyAbstractMemoryBlock mb = Owner.GetAbstractOutput(i);

                    if (mb != null)
                    {
                        mb.SafeCopyToDevice();
                    }
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
        public static void Init(MyCSharpNode owner)
        {
            MyLog.DEBUG.WriteLine(""Init called"");
        }
        
        public static void Execute(MyCSharpNode owner)
        {
            MyLog.DEBUG.WriteLine(""Execute called"");
            
            float[] input = owner.GetInput(0).Host;
            float[] output = owner.GetOutput(0).Host;
        
            output[0] = 3 * (float)Math.Cos(input[0]);
        }
    }
}";
        #endregion
    }
}
