using System;
using System.Runtime.InteropServices;
using OpenTK.Graphics.OpenGL;

namespace Render.RenderObjects.Buffers
{
    internal abstract class VboBase : IDisposable
    {
        public int Handle { get; private set; }
        public int Count { get; private set; }

        public BufferTarget Target { get; private set; }
        public int ElementSize { get; private set; }


        #region Genesis

        protected VboBase(int elementSize = 4, BufferTarget target = BufferTarget.ArrayBuffer)
        {
            Target = target;
            ElementSize = elementSize;
        }

        public virtual void Dispose()
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

        public virtual void Update<T>(T[] data, int count = -1, int offset = 0)
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

    internal class Vbo<T> : VboBase
        where T : struct
    {
        public Vbo(int count, T[] initData = null, int elementSize = 4, BufferTarget target = BufferTarget.ArrayBuffer, BufferUsageHint hint = BufferUsageHint.DynamicDraw)
            : base(elementSize, target)
        {
            Init(count, initData, hint);
        }
    }

    internal sealed class StaticVbo<T> : Vbo<T>
        where T : struct
    {
        public StaticVbo(int count, T[] initData = null, int elementSize = 4, BufferTarget target = BufferTarget.ArrayBuffer, BufferUsageHint hint = BufferUsageHint.DynamicDraw)
            : base(count, initData, elementSize, target, hint)
        { }

        // These are shared by many geometries, prevent their disposal
        public override void Dispose()
        { }


        public override void Update<T1>(T1[] data, int count = -1, int offset = 0)
        {
            // TODO: Allow it for some global changes in the future?
            throw new NotImplementedException("Static data shan't be updated.");
        }
    }
}
