using System;
using System.Runtime.InteropServices;
using OpenTK.Graphics.OpenGL;

namespace Render.RenderObjects.Buffers
{
    internal abstract class VBOBase : IDisposable
    {
        public int Handle { get; private set; }
        public int Count { get; private set; }

        public BufferTarget Target { get; private set; }
        public int ElementSize { get; private set; }


        #region Genesis

        protected VBOBase(int elementSize = 4, BufferTarget target = BufferTarget.ArrayBuffer)
        {
            Target = target;
            ElementSize = elementSize;
        }

        public void Dispose()
        {
            GL.DeleteBuffer(Handle);
        }

        protected void Init<T>(int count, T[] data, BufferUsageHint hint)
            where T : struct
        {
            Handle = GL.GenBuffer();
            Count = count;

            Bind();
            GL.BufferData(Target, count * Marshal.SizeOf(typeof(T)), data, hint);
            Unbind();
        }

        #endregion

        public void Update<T>(T[] data, int count = -1, int offset = 0)
            where T : struct
        {
            if (count == -1)
                count = Math.Min(Count, data.Length);

            var tSize = Marshal.SizeOf(typeof(T));

            Bind();
            GL.BufferSubData(Target, new IntPtr(offset * tSize), count * tSize, data);
            Unbind();
        }

        public void Bind()
        {
            GL.BindBuffer(Target, Handle);
        }

        public void Unbind()
        {
            GL.BindBuffer(Target, 0);
        }
    }

    internal class VBO<T> : VBOBase
        where T : struct
    {
        public VBO(int count, T[] initData = null, int elementSize = 4, BufferTarget target = BufferTarget.ArrayBuffer, BufferUsageHint hint = BufferUsageHint.DynamicDraw)
            : base(elementSize, target)
        {
            Init(count, initData, hint);
        }
    }
}
