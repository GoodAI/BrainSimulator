using System;
using System.Collections.Generic;
using System.Linq;
using GoodAI.ToyWorld.Control;
using Render.RenderRequests.AvatarRenderRequests;
using Render.RenderRequests.RenderRequests;
using Render.Tests.RRs;
using Utils.VRageRIP.Lib.Collections;

namespace Render.RenderRequests
{
    // NOTE: All renderRequests must inherit from RenderRequest
    public static class RenderRequestFactory
    {
        private static readonly TypeSwitch<IRenderRequest> RRSwitch = new TypeSwitch<IRenderRequest>();
        public static IEnumerable<IRenderRequest> RRs { get { return RRSwitch.Matches.Values.Select(rr => rr()); } }

        private static readonly TypeSwitchParam<IAvatarRenderRequest, int> ARRSwitch = new TypeSwitchParam<IAvatarRenderRequest, int>();
        public static IEnumerable<IAvatarRenderRequest> ARRs { get { return ARRSwitch.Matches.Values.Select(rr => rr(0)); } }


        static RenderRequestFactory()
        {
            //////////////////////
            // NOTE: All renderRequests must inherit from RenderRequest
            //////////////////////

            // RenderRequests
            RRSwitch
                .Case<IBasicTexRR>(() =>
                    new BasicTexRR())
                .Case<IFullMapRenderRequest>(() =>
                    new FullMapRR());

            // AvatarRenderRequests
            ARRSwitch
                .Case<IBasicARR>(id =>
                    new BasicARR(id))
                .Case<IFovAvatarRenderRequest>(id =>
                    new FoVARR(id));
        }


        // TODO: log when returning null

        public static T CreateRenderRequest<T>()
            where T : class, IRenderRequest // unf cannot constrain T to be an interface, only a class
        {
            return RRSwitch.Switch<T>();
        }

        public static T CreateRenderRequest<T>(int avatarID)
            where T : class, IAvatarRenderRequest // unf cannot constrain T to be an interface, only a class
        {
            return ARRSwitch.Switch<T>(avatarID);
        }
    }
}
