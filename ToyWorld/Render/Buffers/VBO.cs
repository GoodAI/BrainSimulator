using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using OpenTK.Graphics.OpenGL;

namespace Render.Buffers
{
    internal class VBO<T> : IDisposable
        where T : struct
    {
        private uint m_handle;
        private int count;

        public VBO(int count, T[] initData = null, BufferUsageHint hint = BufferUsageHint.DynamicDraw)
        {
            if (initData != null)
                Debug.Assert(initData.Length == count);

            m_handle = (uint)GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, m_handle);
            GL.BufferData(BufferTarget.ArrayBuffer, count * Marshal.SizeOf(typeof(T)), initData, hint);
            GL.BindBuffer(BufferTarget.ArrayBuffer, 0);
        }

        public void Dispose()
        {
            GL.DeleteBuffer(m_handle);
        }
    }
}
