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
    public class MyCSharpNode : MyScriptableNode, IMyVariableBranchViewNodeBase
    {
        public MyCSharpNode()
        {
            InputBranches = 1;
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

        public override void Validate(MyValidator validator)
        {

        }

        #region inputs & outputs
        [ReadOnly(false)]
        [YAXSerializableField, YAXElementFor("IO")]
        public override int InputBranches
        {
            get { return base.InputBranches; }
            set
            {
                base.InputBranches = Math.Max(value, 1);
            }
        }        

        public int Input0Count { get { return GetInput(0) != null ? GetInput(0).Count : 0; } }
        public int Input0ColHint { get { return GetInput(0) != null ? GetInput(0).ColumnHint : 0; } }

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
            int[] branchSizes = null;

            bool ok = true;
            if (OutputBranchesSpec != null && OutputBranchesSpec != "")
            {
                string[] branchConf = OutputBranchesSpec.Split(',');

                if (branchConf.Length > 0)
                {
                    branchSizes = new int[branchConf.Length];

                    for (int i = 0; i < branchConf.Length; i++)
                    {
                        try
                        {
                            branchSizes[i] = int.Parse(branchConf[i], CultureInfo.InvariantCulture);
                        }
                        catch
                        {
                            ok = false;
                        }
                    }
                }
            }
            if (!ok)
            {
                return null;
            }

            return branchSizes;
        }

        private void UpdateOutputBlocks()
        {
            int [] op = GetOutputBranchSpec();

            if (op != null)
            {
                int sum = 0;
                for (int i = 0; i < op.Length; i++)
                {
                    sum += op[i];
                }

                for (int i = 0; i < op.Length; i++)
                {
                    GetOutput(i).Count = op[i];
                }
            }
        }

        #endregion       

        #region Tasks

        public MyInitScriptTask InitScript { get; private set; }

        [Description("Init script"), MyTaskInfo(OneShot = true)]
        public class MyInitScriptTask : MyTask<MyCSharpNode>
        {
            internal MethodInfo ScriptInitMethod { get; private set; }
            internal MethodInfo ScriptExecuteMethod { get; private set; }

            public override void Init(int nGPU)
            {
                CSharpCodeProvider codeProvider = new CSharpCodeProvider();
                CompilerParameters parameters = new CompilerParameters()
                {
                    GenerateInMemory = false,
                    GenerateExecutable = false,                    
                };

                parameters.ReferencedAssemblies.Add("GoodAI.Platform.Core.dll");
                parameters.ReferencedAssemblies.Add(Assembly.GetExecutingAssembly().Location);

                CompilerResults results = codeProvider.CompileAssemblyFromSource(parameters, Owner.Script);
                Assembly compiledAssembly = null;

                if (results.Errors.HasErrors)
                {
                    MyLog.WARNING.WriteLine(Owner.Name +  ": Errors in compiled script");

                    foreach (CompilerError error in results.Errors)
                    {
                        MyLog.WARNING.WriteLine("Error (" + error.ErrorNumber + "): " + error.ErrorText);
                    }                   
                }
                else
                {
                    compiledAssembly = results.CompiledAssembly;                    
                }

                ScriptInitMethod = null;
                ScriptExecuteMethod = null;

                if (compiledAssembly != null)
                {
                    try
                    {
                        Type eclosingType = compiledAssembly.GetType("Runtime.Script");
                        ScriptInitMethod = eclosingType.GetMethod("Init");
                    }
                    catch (Exception e)
                    {
                        MyLog.WARNING.WriteLine(Owner.Name +  ": Init() method retrieval failed: " + e.GetType().Name + ": " + e.Message);                        
                    }

                    try
                    {
                        Type eclosingType = compiledAssembly.GetType("Runtime.Script");
                        ScriptExecuteMethod = eclosingType.GetMethod("Execute");
                    }
                    catch (Exception e)
                    {
                        MyLog.WARNING.WriteLine(Owner.Name +  ": Execute() method retrieval failed: " + e.GetType().Name + ": " + e.Message);                        
                    }
                }         
            }

            public override void Execute()
            {
                if (ScriptInitMethod != null)
                {
                    for (int i = 0; i < Owner.InputBranches; i++)
                    {
                        Owner.GetAbstractInput(i).SafeCopyToHost();
                    }

                    try
                    {
                        ScriptInitMethod.Invoke(null, new object[] { Owner });
                    }
                    catch (Exception e)
                    {
                        MyLog.WARNING.WriteLine("Script Init() call failed: " + e.GetType().Name + ": " + e.Message);
                    }

                    for (int i = 0; i < Owner.OutputBranches; i++)
                    {
                        Owner.GetAbstractOutput(i).SafeCopyToDevice();
                    }
                }
                else
                {
                    MyLog.WARNING.WriteLine(Owner.Name + ": No Init() method available");
                }
            }
        }
        
        public MyExecuteScriptTask ExecuteScript { get; private set; }     

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
                    Owner.GetAbstractInput(i).SafeCopyToHost();
                }

                if (Owner.InitScript.ScriptExecuteMethod != null)
                {
                    try
                    {
                        Owner.InitScript.ScriptExecuteMethod.Invoke(null, new object[] { Owner });
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
                    Owner.GetAbstractOutput(i).SafeCopyToDevice();
                }
            }
        }

        #endregion
    }
}
