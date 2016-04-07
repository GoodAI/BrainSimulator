using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using Render.Geometries.Buffers;
using Render.RenderObjects.Buffers;
using VRageMath;

namespace Render.RenderObjects.Geometries
{
    internal abstract class GeometryBase : IDisposable
    {
        protected readonly VAO Vao = new VAO();


        public void Dispose()
        {
            Vao.Dispose();
        }


        public abstract void Draw();

        public void Update<T>(string id, T[] data, int count = -1, int offset = 0)
            where T : struct
        {
            Vao[id].Update(data, count, offset);
        }
    }
}
