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

        private void Init<T>(int count, T[] initData, BufferUsageHint hint)
        where T : struct
        {
            if (initData != null)
                Debug.Assert(initData.Length == count);

            Handle = GL.GenBuffer();
            Count = count;

            GL.BindBuffer(BufferTarget.ArrayBuffer, Handle);
            GL.BufferData(BufferTarget.ArrayBuffer, count * Marshal.SizeOf(typeof(T)), initData, hint);
            GL.BindBuffer(BufferTarget.ArrayBuffer, 0);
        }

        public void Dispose()
        {
            GL.DeleteBuffer(Handle);
        }

        #endregion


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
