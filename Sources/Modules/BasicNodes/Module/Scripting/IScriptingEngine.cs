using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Modules.Scripting
{
    public enum MyDefaultMethods
    {
        Init,
        Execute
    }

    //TODO: find better enum constrain
    public interface IScriptingEngine<E> where E : struct
    {
        void Run(E methodName, params object[] arguments);
        bool HasMethod(E methodName);
        void Compile(MyValidator validator);
    }
}
