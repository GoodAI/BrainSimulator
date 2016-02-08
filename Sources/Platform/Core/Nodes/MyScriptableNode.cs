using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using GoodAI.Platform.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Core.Nodes
{
    public interface IScriptableNode : IValidatable
    {
        event EventHandler<MyPropertyChangedEventArgs<string>> ScriptChanged;

        string Script { get; set; }
        string Name { get; }

        /// <summary>
        /// Should return alphabetically ordered space delimited list of name expressions for auto complete & syntax highlighting.
        /// </summary>
        string NameExpressions { get; }

        /// <summary>
        /// Should return alphabetically ordered space delimited list of keywords for auto complete & syntax highlighting.
        /// </summary>
        string Keywords { get; }

        /// <summary>
        /// Should return language name. Temporaly used for syntax highlighting settings.
        /// </summary>
        string Language { get; }
    }

    public abstract class MyScriptableNode : MyWorkingNode, IScriptableNode
    {
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

        public abstract string NameExpressions { get; }
        public abstract string Keywords { get; }
        public abstract string Language { get; }

        protected void CopyInputBlocksToHost()
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

        protected void CopyOutputBlocksToDevice()
        {
            for (int i = 0; i < OutputBranches; i++)
            {
                MyAbstractMemoryBlock mb = GetAbstractOutput(i);

                if (mb != null)
                {
                    mb.SafeCopyToDevice();
                }
            }
        }
    } 
}
