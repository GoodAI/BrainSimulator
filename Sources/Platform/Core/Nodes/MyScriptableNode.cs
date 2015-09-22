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
    public abstract class MyScriptableNode : MyWorkingNode
    {
        [YAXSerializableField]
        protected string m_script;

        protected event EventHandler<MyPropertyChangedEventArgs<string>> ScriptChanged;

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

        /// <summary>
        /// Should return alphabetically ordered space delimited list of name expressions for auto complete & syntax highlighting.
        /// </summary>
        public abstract string NameExpressions { get; }

        /// <summary>
        /// Should return alphabetically ordered space delimited list of keywords for auto complete & syntax highlighting.
        /// </summary>
        public abstract string Keywords { get; }
    } 
}
