using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Game;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
using Render.Renderer;
using Render.RenderRequests;
using Render.Tests.RRs;
using ToyWorldTests.Game;
using Xunit;

namespace ToyWorldTests.Render
{
    [Collection("Renderer")]
    public class RenderRequestTests : GameControllerTestBase
    {
        [Fact]
        public void RRInits()
        {
            Assert.NotNull(GameController.Renderer);
            GameController.Renderer.MakeContextCurrent();

            foreach (var rr in RenderRequestFactory.RRs)
            {
                var r = rr as RenderRequest;
                Assert.NotNull(r);
                r.Init(GameController.Renderer, GameController.World);
                GameController.Renderer.CheckError();
            }

            foreach (var rr in RenderRequestFactory.AvatarRRs)
            {
                var r = rr as RenderRequest;
                Assert.NotNull(r);
                r.Init(GameController.Renderer, GameController.World);
                GameController.Renderer.CheckError();
            }

            GameController.Renderer.MakeContextNotCurrent();
        }

        [Fact]
        public void GatherImageTest()
        {
            var RRTest = GameController.RegisterRenderRequest<IFullMapRR>();

            Assert.True(RRTest.Image == null || RRTest.Image.Length == 0);

            GameController.MakeStep();
            Assert.True(
                RRTest.Image == null
                || RRTest.Image.Length == 0
                || RRTest.Image.All(u => u == 0));

            RRTest.GatherImage = true;
            GameController.MakeStep();
            Assert.True(RRTest.Image.Length >= RRTest.Resolution.Width * RRTest.Resolution.Height); // Allocation needn't be immediate
            Assert.Contains(RRTest.Image, u => (u & 0xFFFFFF00) != 0);

            RRTest.GatherImage = false;
            GameController.MakeStep();
            Assert.True(
                RRTest.Image == null
                || RRTest.Image.Length == 0
                || RRTest.Image.All(u => u == 0));
        }

        [Fact]
        public void ChangeResolutionTest()
        {
            var RRTest = GameController.RegisterRenderRequest<IFullMapRR>();
            RRTest.GatherImage = true;

            GameController.MakeStep();
            Assert.Contains(RRTest.Image, u => (u & 0xFFFFFF00) != 0);

            RRTest.Resolution = new Size(32, 32);
            GameController.MakeStep();
            Assert.Contains(RRTest.Image, u => (u & 0xFFFFFF00) != 0);

            RRTest.Resolution = new Size(2048, 1280);
            GameController.MakeStep();
            Assert.Contains(RRTest.Image, u => (u & 0xFFFFFF00) != 0);
        }

        [Fact]
        public void RenderRequestThrows()
        {
            var RRTest = GameController.RegisterRenderRequest<IFullMapRR>();

            Assert.ThrowsAny<ArgumentOutOfRangeException>(() => RRTest.Resolution = Size.Empty);
            Assert.ThrowsAny<ArgumentOutOfRangeException>(() => RRTest.Resolution = new Size(65536, 65536));
        }


        #region RRs

        private void TestStep<T>()
            where T : class, IRenderRequest
        {
            var RRTest = GameController.RegisterRenderRequest<T>();
            GameController.MakeStep();
        }

        [Fact]
        public void FullMapRR()
        {
            TestStep<IFullMapRR>();
        }

        [Fact]
        public void FreeMapRR()
        {
            TestStep<IFreeMapRR>();
        }

        #endregion

        #region AvatarRRs

        private const int AvatarID = 1;

        private void TestStepAvatar<T>(Action<T> rrSetupAction = null)
            where T : class, IAvatarRenderRequest
        {
            var RRTest = GameController.RegisterRenderRequest<T>(AvatarID);

            if (rrSetupAction != null)
                rrSetupAction(RRTest);

            GameController.MakeStep();
        }

        [Fact]
        public void FovAvatarRR()
        {
            TestStepAvatar<IFovAvatarRR>();
        }

        [Fact]
        public void FofAvatarRR()
        {
            TestStepAvatar<IFofAvatarRR>(rr => rr.FovAvatarRenderRequest = GameController.RegisterRenderRequest<IFovAvatarRR>(AvatarID));
        }

        [Fact]
        public void FoFAvatarRRThrows()
        {
            var RRTest = GameController.RegisterRenderRequest<IFofAvatarRR>(AvatarID);
            Assert.ThrowsAny<MissingFieldException>((Action)GameController.MakeStep);
            Assert.ThrowsAny<ArgumentException>(() => RRTest.FovAvatarRenderRequest = null);

            var RR = GameController.RegisterRenderRequest<IFovAvatarRR>(AvatarID);
            RR.Size = new SizeF(1, 1);
            Assert.ThrowsAny<ArgumentException>(() => RRTest.FovAvatarRenderRequest = RR);

            //var differentRR = GameController.RegisterRenderRequest<IFovAvatarRR>(0);
            //Assert.ThrowsAny<ArgumentException>(() => RRTest.FovAvatarRenderRequest = differentRR); // TODO: need at least two avatars for this test
        }

        #endregion
    }
}
