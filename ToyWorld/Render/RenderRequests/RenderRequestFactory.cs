using System;
using System.Collections.Generic;
using GoodAI.ToyWorld.Control;
using Render.RenderRequests.RenderRequests;

namespace Render.RenderRequests
{
    // NOTE: All renderRequests must inherit from RenderRequest
    public static class RenderRequestFactory
    {
        private static readonly TypeSwitch<IRenderRequest> RRSwitch = new TypeSwitch<IRenderRequest>();
        private static readonly TypeSwitchParam<IAgentRenderRequest> ARRSwitch = new TypeSwitchParam<IAgentRenderRequest>();


        static RenderRequestFactory()
        {
            RRSwitch.Case<IRRTest>(() => new RRTest());


            //ARRSwitch.Case(() => null);
        }


        // TODO: log when returning null

        public static T CreateRenderRequest<T>()
            where T : class, IRenderRequest // unf cannot constrain T to be an interface, only a class
        {
            return RRSwitch.Switch<T>();
        }

        public static T CreateAgentRenderRequest<T>(int agentID)
            where T : class, IAgentRenderRequest // unf cannot constrain T to be an interface, only a class
        {
            return ARRSwitch.Switch<T>(agentID);
        }


        #region TypeSwitch

        internal abstract class TypeSwitchBase<TKey, TVal>
            where TVal : class
        {
            private readonly Dictionary<Type, TVal> m_matches = new Dictionary<Type, TVal>();


            public TypeSwitchBase<TKey, TVal> Case<T>(TVal action)
                where T : class, TKey
            {
                m_matches.Add(typeof(T), action);
                return this;
            }

            protected TVal SwitchInternal<T>()
                where T : class, TKey
            {
                TVal res;

                if (!m_matches.TryGetValue(typeof(T), out res))
                {
                    // log
                    return null;
                }

                return res;
            }
        }

        private class TypeSwitch<TKey> : TypeSwitchBase<TKey, Func<TKey>>
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

        private class TypeSwitchParam<TKey> : TypeSwitchBase<TKey, Func<int, TKey>>
        {
            public TRet Switch<TRet>(int id)
                where TRet : class, TKey
            {
                var res = SwitchInternal<TRet>();

                if (res != null)
                    return (TRet)res(id);

                return null;
            }
        }

        #endregion
    }
}
