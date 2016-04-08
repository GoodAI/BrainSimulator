using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Buffers;
using VRageMath;

namespace Render.RenderObjects.Geometries
{
    internal abstract class GeometryBase : VAO
    {
        public abstract void Draw();

        public void Update<T>(string id, T[] data, int count = -1, int offset = 0)
            where T : struct
        {
            this[id].Update(data, count, offset);
        }
    }
}
