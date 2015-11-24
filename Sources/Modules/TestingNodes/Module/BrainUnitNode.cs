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
using YAXLib;

// TODO: resolve duplicate code, this is mostly copy of the CSharpNode
namespace GoodAI.Modules.TestingNodes
{
    /// <author>GoodAI</author>
    /// <status>Working</status>
    /// <summary>Experimental C# scripting node</summary>
    /// <description>Node allows user to write C# code directly and run it during simulation. Example code:
    /// TODO
    ///
    /// </description>    
    public class BrainUnitNode : MyScriptableNode, IMyVariableBranchViewNodeBase
    {
        public BrainUnitNode()
        {
            InputBranches = 1;
            Script = EXAMPLE_CODE;
        }

        public override string NameExpressions
        {
            get { return "GetInput"; }
        }

        public override string Keywords
        {
            get
            {
                return
                    "abstract as base break case catch checked continue default delegate do else event explicit extern false finally fixed for foreach goto if implicit in interface internal is lock namespace new null object operator out override params private protected public readonly ref return sealed sizeof stackalloc switch this throw true try typeof unchecked unsafe using virtual while"
                    + " bool byte char class const decimal double enum float int long sbyte short static string struct uint ulong ushort void";
            }
        }

        public override string Language
        {
            get { return "CSharp"; }
        }

        public override void UpdateMemoryBlocks()
        {
        }

        #region Brain Unit

        [MyBrowsable, Category("BrainUnit")]
        [YAXSerializableField(DefaultValue = 1)]
        public int MaxStepCount { get; set; }

        public bool IsUnderTest { get; private set; }

        public void Check()
        {
            CopyInputBlocksToHost();

            if (ScriptCheckMethod == null)
            {
                MyLog.WARNING.WriteLine(Name + ": No Check() method available");
                return;
            }

            try
            {
                ScriptCheckMethod.Invoke(null, new object[] {this});
            }
            catch (TargetInvocationException e)
            {
                Exception innerException = e.InnerException ?? e;
                MyLog.DEBUG.WriteLine("Exception occurred inside the Check(): " + innerException.Message);
                throw innerException; // allow to catch assert failures
            }
            catch (Exception e)
            {
                MyLog.WARNING.WriteLine("Script Check() call failed: " + e.GetType().Name + ": " + e.Message);
            }
        }

        private void CopyInputBlocksToHost()
        {
            for (int i = 0; i < InputBranches; i++)
            {
                MyAbstractMemoryBlock mb = GetAbstractInput(i);

                if (mb != null)
                {
                    mb.SafeCopyToHost();
                }
            }
        }

        #endregion

        #region inputs
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

        #endregion

        #region Compilation

        internal MethodInfo ScriptInitMethod { get; private set; }
        internal MethodInfo ScriptCheckMethod { get; private set; }

        public override void Validate(MyValidator validator)
        {
            ScriptInitMethod = null;
            ScriptCheckMethod = null;

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

                    ScriptCheckMethod = enclosingType.GetMethod("Check");
                    validator.AssertError(ScriptCheckMethod != null, this, "Check() method not found in compiled script");
                }
                catch (Exception e)
                {
                    validator.AddError(this, "Script analysis failed: " + e.GetType().Name + ": " + e.Message);
                }
            }
        }
        #endregion

        #region ExampleCode
        private const string EXAMPLE_CODE = @"using System;
using GoodAI.Core.Utils;
using GoodAI.Core.Nodes;
using GoodAI.Modules.TestingNodes;

namespace Runtime
{
    public class Script
    {
        public static void Check(BrainUnitNode owner)
        {
            MyLog.DEBUG.WriteLine(""Check called"");
            
            float[] input = owner.GetInput(0).Host;
        }
    }
}";
        #endregion
    }
}
