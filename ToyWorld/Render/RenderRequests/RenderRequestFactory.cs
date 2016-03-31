using System;
using System.Collections.Generic;
using System.Linq;
using GoodAI.ToyWorld.Control;
using Render.RenderRequests.AvatarRenderRequests;
using Render.RenderRequests.RenderRequests;
using Render.RenderRequests.Tests;

namespace Render.RenderRequests
{
    // NOTE: All renderRequests must inherit from RenderRequest
    public static class RenderRequestFactory
    {
        private static readonly TypeSwitch<IRenderRequest> RRSwitch = new TypeSwitch<IRenderRequest>();
        public static IEnumerable<IRenderRequest> RRs { get { return RRSwitch.Matches.Values.Select(rr => rr()); } }

        private static readonly TypeSwitchParam<IAvatarRenderRequest> ARRSwitch = new TypeSwitchParam<IAvatarRenderRequest>();
        public static IEnumerable<IAvatarRenderRequest> ARRs { get { return ARRSwitch.Matches.Values.Select(rr => rr(0)); } }


        static RenderRequestFactory()
        {
            // RenderRequests
            RRSwitch.Case<IRRTest>(() => new RRTest());

            // AvatarRenderRequests
            ARRSwitch.Case<IARRTest>(id => new ARRTest(id));
            ARRSwitch.Case<IAvatarRenderRequestFoV>(id => new ARRFoV(id));
        }


        // TODO: log when returning null

        public static T CreateRenderRequest<T>()
            where T : class, IRenderRequest // unf cannot constrain T to be an interface, only a class
        {
            return RRSwitch.Switch<T>();
        }

        public static T CreateAvatarRenderRequest<T>(int avatarID)
            where T : class, IAvatarRenderRequest // unf cannot constrain T to be an interface, only a class
        {
            return ARRSwitch.Switch<T>(avatarID);
        }


        #region TypeSwitch

        internal abstract class TypeSwitchBase<TKey, TVal>
            where TVal : class
        {
            internal readonly Dictionary<Type, TVal> Matches = new Dictionary<Type, TVal>();


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

        private sealed class TypeSwitch<TKey> : TypeSwitchBase<TKey, Func<TKey>>
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

        private sealed class TypeSwitchParam<TKey> : TypeSwitchBase<TKey, Func<int, TKey>>
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
