using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Modules.Scripting
{
    public enum DefaultMethods
    {
        Init,
        Execute
    }

    //TODO: find better enum constrain
    public interface IScriptingEngine<TMethodEnum> where TMethodEnum : struct
    {
        void Run(TMethodEnum methodName, params object[] arguments);
        bool HasMethod(TMethodEnum methodName);
        void Compile(MyValidator validator);
        
        /// <summary>
        /// Should return alphabetically ordered space delimited list of keywords for auto complete & syntax highlighting.
        /// </summary>
        string DefaultKeywords { get; }

        /// <summary>
        /// Should return supported language
        /// </summary>
        string Language { get; }
    }
}
