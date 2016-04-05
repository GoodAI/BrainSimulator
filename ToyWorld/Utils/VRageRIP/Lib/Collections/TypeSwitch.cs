using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Utils.VRageRIP.Lib.Collections
{
    public abstract class TypeSwitchBase<TKey, TVal>
        where TVal : class
    {
        public Dictionary<Type, TVal> Matches { get; private set; }


        protected TypeSwitchBase()
        {
            Matches = new Dictionary<Type, TVal>();
        }


        public TypeSwitchBase<TKey, TVal> Case<T>(TVal action)
            where T : class, TKey
        {
            Matches.Add(typeof(T), action);
            return this;
        }

        protected TVal SwitchInternal<T>()
            where T : class, TKey
        {
            TVal res;

            if (!Matches.TryGetValue(typeof(T), out res))
            {
                // log
                return null;
            }

            return res;
        }
    }

    public sealed class TypeSwitch<TKey> : TypeSwitchBase<TKey, Func<TKey>>
    {
        public TRet Switch<TRet>()
            where TRet : class, TKey
        {
            var res = SwitchInternal<TRet>();

            if (res != null)
                return (TRet)res();

            return null;
        }
    }

    public sealed class TypeSwitchParam<TKey, TParam> : TypeSwitchBase<TKey, Func<TParam, TKey>>
    {
        public TRet Switch<TRet>(TParam id)
            where TRet : class, TKey
        {
            var res = SwitchInternal<TRet>();

            if (res != null)
                return (TRet)res(id);

            return null;
        }
    }

}
