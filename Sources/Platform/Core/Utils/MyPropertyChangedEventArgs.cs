using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Platform.Core.Utils
{
    public class MyPropertyChangedEventArgs<T> 
    {
        public T OldValue { get; private set; }
        public T NewValue { get; private set; }

        public MyPropertyChangedEventArgs(T oldValue, T newValue)
        {
            OldValue = oldValue;
            NewValue = newValue;
        }
    }
}
