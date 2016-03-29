using System;
using System.Collections.Generic;
using GoodAI.ToyWorld.Control;
using Render.Renderer;

namespace Render.RenderRequests.Setup
{
    public static class RenderRequestFactory
    {
        private static readonly TypeSwitch<IRenderRequest> RRSwitch = new TypeSwitch<IRenderRequest>();
        private static readonly TypeSwitch<IAgentRenderRequest> ARRSwitch = new TypeSwitch<IAgentRenderRequest>();


        static RenderRequestFactory()
        {
            RRSwitch.Case(CreateRenderRequestTest);


            ARRSwitch.Case(CreateAgentRenderRequestFoV);
        }


        public static T CreateRenderRequest<T>()
            where T : class, IRenderRequest // unf cannot constrain T to be an interface, only a class
        {
            return RRSwitch.Switch<T>();
        }

        public static T CreateAgentRenderRequest<T>()
            where T : class, IAgentRenderRequest // unf cannot constrain T to be an interface, only a class
        {
            return ARRSwitch.Switch<T>();
        }


        #region IrenderRequests

        private static IRenderRequestTest CreateRenderRequestTest()
        {
            return new RenderRequestTest();
        }

        #endregion

        #region IAgentRenderRequests

        private static IAgentRenderRequest CreateAgentRenderRequestFoV()
        {
            return null;
        }

        #endregion
    }


    internal class TypeSwitch<TBase>
    {
        private readonly Dictionary<Type, Func<TBase>> m_matches = new Dictionary<Type, Func<TBase>>();


        public TypeSwitch<TBase> Case<T>(Func<T> action)
            where T : class, TBase
        {
            m_matches.Add(typeof (T), action);
            return this;
        }

        public T Switch<T>()
            where T : class, TBase
        {
            Func<TBase> res;

            if (!m_matches.TryGetValue(typeof (T), out res))
            {
                // log
                return null;
            }

            return ((Func<T>) res)();
        }
    }
}