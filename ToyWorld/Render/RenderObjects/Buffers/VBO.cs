using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenTK.Graphics.OpenGL;

namespace Render.Geometries.Buffers
{
    internal class VBO
    {
        public uint Handle { get; private set; }
        public int Count { get; set; }

        private readonly BufferTarget m_target;


        #region Creation/destruction

        public VBO(int count, float[] initData = null, BufferTarget target = BufferTarget.ArrayBuffer, BufferUsageHint hint = BufferUsageHint.DynamicDraw)
        {
            m_target = target;
            Init(count, initData, hint);
        }

        public VBO(int count, uint[] initData = null, BufferTarget target = BufferTarget.ArrayBuffer, BufferUsageHint hint = BufferUsageHint.DynamicDraw)
        {
            m_target = target;
            Init(count, initData, hint);
        }

        private void Init<T>(int count, T[] initData, BufferUsageHint hint)
        where T : struct
        {
            if (initData != null)
                Debug.Assert(initData.Length == count);

            Handle = (uint)GL.GenBuffer();
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
            GL.BindBuffer(m_target, Handle);
        }

        public void Unbind()
        {
            GL.BindBuffer(m_target, Handle);
        }
    }
}
