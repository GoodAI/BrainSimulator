﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Utils.VRageRIP.Lib.Collections
{
    public abstract class TypeSwitchBase<TKeyBase, TValBase>
        where TValBase : class
    {
        public Dictionary<Type, TValBase> Matches { get; private set; }


        protected TypeSwitchBase()
        {
            Matches = new Dictionary<Type, TValBase>();
        }


        public TypeSwitchBase<TKeyBase, TValBase> Case<TKey>(TValBase action)
            where TKey : class, TKeyBase
        {
            Matches.Add(typeof(TKey), action);
            return this;
        }

        protected TValBase SwitchInternal<TKey>()
            where TKey : class, TKeyBase
        {
            TValBase res;

            if (!Matches.TryGetValue(typeof(TKey), out res))
            {
                // log
                return null;
            }

            return res;
        }
    }

    public sealed class TypeSwitch<TKeyBase> : TypeSwitchBase<TKeyBase, Func<TKeyBase>>
    {
        public TRet Switch<TRet>()
            where TRet : class, TKeyBase
        {
            var res = SwitchInternal<TRet>();

            if (res != null)
                return (TRet)res();

            return null;
        }
    }

    public sealed class TypeSwitchParam<TKeyBase, TParam> : TypeSwitchBase<TKeyBase, Func<TParam, TKeyBase>>
    {
        public TRet Switch<TRet>(TParam par)
            where TRet : class, TKeyBase
        {
            var res = SwitchInternal<TRet>();

            if (res != null)
                return (TRet)res(par);

            return null;
        }
    }

    public sealed class TypeSwitchParam<TKeyBase, TParam1, TParam2> : TypeSwitchBase<TKeyBase, Func<TParam1, TParam2, TKeyBase>>
    {
        public TRet Switch<TRet>(TParam1 par1, TParam2 par2)
            where TRet : class, TKeyBase
        {
            var res = SwitchInternal<TRet>();

            if (res != null)
                return (TRet)res(par1, par2);

            return null;
        }
    }
}
