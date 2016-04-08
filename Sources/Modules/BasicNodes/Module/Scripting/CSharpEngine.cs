using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Core.Configuration;
using Microsoft.CSharp;
using System;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace GoodAI.Modules.Scripting
{
    public class CSharpEngine<TMethodEnum> : IScriptingEngine<TMethodEnum> where TMethodEnum : struct
    {
        private Dictionary<string, MethodInfo> m_methods;
        private IScriptableNode m_node;

        public CSharpEngine(IScriptableNode node)
        {
            if (!typeof(TMethodEnum).IsEnum)
            {
                throw new ArgumentException("Only enum types allowed for method enumeration:" + typeof(TMethodEnum).Name);
            }

            m_methods = new Dictionary<string, MethodInfo>();
            m_node = node;
        }
        
        public void Run(TMethodEnum methodName, params object[] arguments)
        {
            MethodInfo method = null;

            if (m_methods.TryGetValue(methodName.ToString(), out method))
            {
                method.Invoke(null, arguments);
            }
        }

        public bool HasMethod(TMethodEnum methodName)
        {
            return m_methods.ContainsKey(methodName.ToString());
        }

        public void Compile(MyValidator validator)
        {
            Compile(m_node.Script, validator);
        }

        public void Compile(string script, MyValidator validator)
        {
            if (m_node.Language != this.Language)
            {
                throw new ArgumentException("Language is not supported (" + this.Language + " only): " + m_node.Language);
            }

            m_methods.Clear();

            CSharpCodeProvider codeProvider = new CSharpCodeProvider();            
            CompilerParameters parameters = new CompilerParameters()
            {
                GenerateInMemory = false,
                GenerateExecutable = false,
            };

            parameters.ReferencedAssemblies.Add("GoodAI.Platform.Core.dll");
            parameters.ReferencedAssemblies.Add("System.Core.dll"); //for LINQ support

            // add all loaded dll modules to be accessible also from the C# node script
            foreach (FileInfo assemblyFile in MyConfiguration.ListModules())
            {
                parameters.ReferencedAssemblies.Add(assemblyFile.FullName);
            }

            parameters.ReferencedAssemblies.Add(Assembly.GetExecutingAssembly().Location);

            Assembly[] loadedAssemblies = AppDomain.CurrentDomain.GetAssemblies();
            IEnumerable<Assembly> openTKAssemblies = loadedAssemblies.Where(x => x.ManifestModule.Name == "OpenTK.dll").ToList();

            if (openTKAssemblies.Any())
            {
                parameters.ReferencedAssemblies.Add(openTKAssemblies.First().Location);
            }

            CompilerResults results = codeProvider.CompileAssemblyFromSource(parameters, script);
            Assembly compiledAssembly = null;

            if (results.Errors.HasErrors)
            {
                string message = "";

                foreach (CompilerError error in results.Errors)
                {
                    message += "\nLine " + error.Line + ": " + error.ErrorText;
                }
                validator.AddError(m_node, "Errors in compiled script:" + message);
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
                    Type methodEnum = typeof(TMethodEnum);

                    foreach (string methodName in methodEnum.GetEnumNames())
                    {
                        MethodInfo method = enclosingType.GetMethod(methodName);                        

                        if (method != null)
                        {
                            m_methods[methodName] = method;
                        }
                        else
                        {
                            validator.AddError(m_node, methodName + "() method not found in compiled script");
                        }
                    }                    
                }
                catch (Exception e)
                {
                    validator.AddError(m_node, "Script analysis failed: " + e.GetType().Name + ": " + e.Message);
                }
            }
        }

        public string DefaultNameExpressions
        {
            get { return String.Empty; }
        }

        public string DefaultKeywords
        {
            get
            {
                return
                    "abstract as base break case catch checked continue default delegate do else event explicit extern false finally fixed for foreach goto if implicit in interface internal is lock namespace new null object operator out override params private protected public readonly ref return sealed sizeof stackalloc switch this throw true try typeof unchecked unsafe using virtual while"
                    + " bool byte char class const decimal double enum float int long sbyte short static string struct uint ulong ushort void";
            }
        }

        public string Language { get { return "CSharp"; } }
    }
}
