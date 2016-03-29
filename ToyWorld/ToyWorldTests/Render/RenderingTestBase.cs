using System;
using System.Threading;
using System.Threading.Tasks;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
using Render.Renderer;
using Utils;
using World.Tiles;
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
