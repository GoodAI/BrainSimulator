using System;
using System.Threading;
using System.Threading.Tasks;
using Game;
using GoodAI.ToyWorld.Control;
using OpenTK.Input;
using Render.RenderRequests;
using Render.RenderRequests.Tests;
using Xunit;

namespace ToyWorldTests.Render
{
    public class RenderRequestTests
    {
        [Fact(Skip = "Manual input needed")]
        //[Fact]
        public void AvatarFoV()
        {
            var gc = ControllerFactory.GetController() as GameControllerBase;
            gc.Init(null);

            Key winKeypressResult = default(Key);
            gc.Renderer.Window.KeyDown += (sender, args) => winKeypressResult = args.Key;
            gc.Renderer.Window.Visible = true;

            var RRTest = gc.RegisterAvatarRenderRequest<IAvatarRenderRequestFoV>(0);

            while (winKeypressResult == default(Key))
            {
                gc.MakeStep();
                gc.Renderer.Context.SwapBuffers();
            }

            gc.Dispose();


            Assert.Equal(winKeypressResult, Key.A);
        }
    }
}
