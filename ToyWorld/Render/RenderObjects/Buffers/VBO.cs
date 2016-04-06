using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenTK.Graphics.OpenGL;

namespace Render.Geometries.Buffers
{
    internal class VBO
    {
        public int Handle { get; private set; }
        public int Count { get; private set; }

        public BufferTarget Target { get; private set; }
        public int ElementSize { get; private set; }


        #region Creation/destruction

        public VBO(int count, float[] initData = null, int elementSize = 4, BufferTarget target = BufferTarget.ArrayBuffer, BufferUsageHint hint = BufferUsageHint.DynamicDraw)
        {
            Target = target;
            ElementSize = elementSize;
            Init(count, initData, hint);
        }

        //public VBO(int count, uint[] initData = null, int size = 4, BufferTarget target = BufferTarget.ArrayBuffer, BufferUsageHint hint = BufferUsageHint.DynamicDraw)
        //{
        //    Target = target;
        //    Size = size;
        //    Init(count, initData, hint);
        //}

        private void Init<T>(int count, T[] data, BufferUsageHint hint)
        where T : struct
        {
            Handle = GL.GenBuffer();
            Count = count;

            Bind();
            GL.BufferData(Target, count * Marshal.SizeOf(typeof(T)), data, hint);
            Unbind();
        }

        public void Dispose()
        {
            GL.DeleteBuffer(Handle);
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
}
