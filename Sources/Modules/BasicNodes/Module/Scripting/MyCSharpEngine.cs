using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using Microsoft.CSharp;
using System;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Modules.Scripting
{
    public class MyCSharpEngine<E> : IScriptingEngine<E> where E : struct
    {
        private Dictionary<string, MethodInfo> m_methods;
        private IScriptableNode m_node;

        public MyCSharpEngine(IScriptableNode node)
        {
            if (!typeof(E).IsEnum)
            {
                throw new ArgumentException("Only enum types allowed for method enumeration:" + typeof(E).Name);
            }

            m_methods = new Dictionary<string, MethodInfo>();
            m_node = node;
        }
        
        public void Run(E methodName, params object[] arguments)
        {
            MethodInfo method = null;

            if (m_methods.TryGetValue(methodName.ToString(), out method))
            {
                method.Invoke(null, arguments);
            }
        }

        public bool HasMethod(E methodName)
        {
            return m_methods.ContainsKey(methodName.ToString());
        }

        public void Compile(MyValidator validator)
        {
            if (m_node.Language != "CSharp")
            {
                throw new ArgumentException("Language is not supported (CSharp only): " + m_node.Language);
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
            parameters.ReferencedAssemblies.Add(Assembly.GetExecutingAssembly().Location);

            Assembly[] loadedAssemblies = AppDomain.CurrentDomain.GetAssemblies();
            IEnumerable<Assembly> openTKAssemblies = loadedAssemblies.Where(x => x.ManifestModule.Name == "OpenTK.dll");
            if (openTKAssemblies.Count() > 0)
                parameters.ReferencedAssemblies.Add(openTKAssemblies.First().Location);

            CompilerResults results = codeProvider.CompileAssemblyFromSource(parameters, m_node.Script);
            Assembly compiledAssembly = null;

            if (results.Errors.HasErrors)
            {
                string message = "";

                foreach (CompilerError error in results.Errors)
                {
                    message += "Line " + error.Line + ": " + error.ErrorText + "\n";
                }
                validator.AddError(m_node, "Errors in compiled script:\n" + message);
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
                    Type methodEnum = typeof(E);

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
    }
}
