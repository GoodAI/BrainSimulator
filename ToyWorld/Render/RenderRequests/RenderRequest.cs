using System;
using GoodAI.ToyWorld.Control;
using Render.Renderer;
using VRageMath;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    public abstract class RenderRequest : IDisposable
    {
        public virtual void Dispose()
        { }


        public abstract void Init(RendererBase renderer, ToyWorld world);
        public abstract void Draw(RendererBase renderer, ToyWorld world);


        protected Matrix GetViewMatrix(Vector3 rotation, Vector3 translation)
        {
            var viewMatrix = Matrix.Identity;

            if (rotation.X > 0)
                viewMatrix = Matrix.CreateRotationZ(rotation.X);

            if (rotation.Y > 0)
                viewMatrix *= Matrix.CreateRotationX(rotation.Y);

            if (rotation.Z > 0)
                viewMatrix *= Matrix.CreateRotationY(rotation.Z);

            Vector3 tar = new Vector3(translation.X, translation.Y, 0);
            translation.Z = 20;
            viewMatrix *= Matrix.CreateLookAt(translation, tar, viewMatrix.Up);

            return viewMatrix;
        }
    }
}
