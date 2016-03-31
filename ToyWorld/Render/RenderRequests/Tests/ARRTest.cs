using OpenTK;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
using Render.Renderer;
using Render.RenderObjects.Geometries;
using Render.RenderRequests.AvatarRenderRequests;

namespace Render.RenderRequests.Tests
{
    class ARRTest : AvatarRenderRequestBase, IARRTest
    {
        public ARRTest(int avatarID)
            : base(avatarID)
        { }


        #region IARRTest overrides

        public float MemAddress { get; set; }

        #endregion

        #region RenderRequestBase overrides

        public override void Init(IRenderer renderer)
        {
            renderer.Window.Visible = true;
        }

        public override void Draw(RendererBase renderer)
        {
        }

        #endregion
    }
}
