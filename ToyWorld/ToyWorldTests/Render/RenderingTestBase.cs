using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
using Render.Renderer;
using Render.RenderRequests;
using Render.RenderRequests.Tests;
using Utils;
using Xunit;

namespace ToyWorldTests.Render
{
    public class RenderingTestBase
    {
        protected Key ManualDebugDraw(string title = Globals.AppName)
        {
            var r = new GLRenderer();

            r.CreateWindow(title, 1024, 768);
            r.CreateContext();
            r.Init();

            Key result = 0;
            r.Window.KeyDown += (sender, args) => result = args.Key;

            var rr = RenderRequestFactory.CreateRenderRequest<IRRTest>();
            (rr as RenderRequest).Init(r);

            r.EnqueueRequest(rr);

            while (result == 0)
            {
                r.ProcessRequests();
                r.Window.ProcessEvents();
            }

            r.Dispose();

            Assert.True(result == Key.A);
            return result;
        }
    }
}
