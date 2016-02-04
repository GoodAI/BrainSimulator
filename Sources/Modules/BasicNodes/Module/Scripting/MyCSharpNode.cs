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

        internal MethodInfo ScriptInitMethod { get; private set; }
        internal MethodInfo ScriptExecuteMethod { get; private set; }

        //internal MethodInfo ScriptOutputNameGetter { get; private set; }
        //internal MethodInfo ScriptInputNameGetter { get; private set; }

        public override void Validate(MyValidator validator)
        {
            ScriptInitMethod = null;
            ScriptExecuteMethod = null;

            CSharpCodeProvider codeProvider = new CSharpCodeProvider();
            CompilerParameters parameters = new CompilerParameters()
            {
                GenerateInMemory = false,
                GenerateExecutable = false,
            };

            parameters.ReferencedAssemblies.Add("GoodAI.Platform.Core.dll");
            parameters.ReferencedAssemblies.Add("System.Core.dll"); //for LINQ support
            parameters.ReferencedAssemblies.Add(Assembly.GetExecutingAssembly().Location);

            Assembly[] loadedAssemblies = AppDomain.CurrentDomain.GetAssemblies();
            IEnumerable<Assembly> openTKAssemblies = loadedAssemblies.Where(x => x.ManifestModule.Name == "OpenTK.dll");
            if (openTKAssemblies.Count() > 0)
                parameters.ReferencedAssemblies.Add(openTKAssemblies.First().Location);

            CompilerResults results = codeProvider.CompileAssemblyFromSource(parameters, Script);
            Assembly compiledAssembly = null;

            if (results.Errors.HasErrors)
            {
                string message = "";

                foreach (CompilerError error in results.Errors)
                {
                    message += "Line " + error.Line + ": " + error.ErrorText + "\n";
                }
                validator.AddError(this, "Errors in compiled script:\n" + message);                
            }
            else
            {
                compiledAssembly = results.CompiledAssembly;
            }            

            if (compiledAssembly != null)
            {
                try
                {
                    Type enclosingType = compiledAssembly.GetType("Runtime.Script");

                    ScriptInitMethod = enclosingType.GetMethod("Init");
                    validator.AssertError(ScriptInitMethod != null, this, "Init() method not found in compiled script");

                    ScriptExecuteMethod = enclosingType.GetMethod("Execute");
                    validator.AssertError(ScriptExecuteMethod != null, this, "Execute() method not found in compiled script");

                    /*
                    ScriptInputNameGetter = enclosingType.GetMethod("GetInputName");

                    if (!CheckNameGetterMethod(ScriptInputNameGetter)) 
                    {
                        validator.AddWarning(this, "\"string GetInputName(int)\" method not found in compiled script");
                        ScriptInputNameGetter = null;
                    }

                    ScriptOutputNameGetter = enclosingType.GetMethod("GetOutputName");

                    if (!CheckNameGetterMethod(ScriptOutputNameGetter))
                    {
                        validator.AddWarning(this, "\"string GetOutputName(int)\" method not found in compiled script");
                        ScriptOutputNameGetter = null;
                    }
                    */
                }
                catch (Exception e)
                {
                    validator.AddError(this, "Script analysis failed: " + e.GetType().Name + ": " + e.Message);
                }                             
            }
        }

        private bool CheckNameGetterMethod(MethodInfo methodInfo)
        {
            if (methodInfo == null ||
                methodInfo.ReturnParameter.ParameterType != typeof(string) ||
                methodInfo.GetParameters().Length != 1 ||
                methodInfo.GetParameters()[0].ParameterType != typeof(int)) return false;
            else return true;
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
                if (Owner.ScriptInitMethod != null)
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
                        Owner.ScriptInitMethod.Invoke(null, new object[] { Owner });
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

                if (Owner.ScriptExecuteMethod != null)
                {
                    try
                    {
                        Owner.ScriptExecuteMethod.Invoke(null, new object[] { Owner });
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
