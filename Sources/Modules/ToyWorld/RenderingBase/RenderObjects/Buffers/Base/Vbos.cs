using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenTK.Graphics.OpenGL;

namespace RenderingBase.RenderObjects.Buffers
{
    public abstract class VboBase 
        : IDisposable
    {
        public uint Handle { get; private set; }

        public int ByteCount { get; protected set; }

        public BufferTarget Target { get; private set; }
        public int ElementSize { get; private set; }


        #region Genesis

        protected VboBase(int elementSize = 4, BufferTarget target = BufferTarget.ArrayBuffer)
        {
            Handle = (uint)GL.GenBuffer();

            Target = target;
            ElementSize = elementSize;
        }

        public virtual void Dispose()
        {
            GL.DeleteBuffer(Handle);
        }

        #endregion


        public void Bind()
        {
            Bind(Target);
        }

        public void Bind(BufferTarget target)
        {
            GL.BindBuffer(target, Handle);
        }

        //public void Unbind()
        //{
        //    GL.BindBuffer(Target, 0);
        //}
    }

    public abstract class VboBase<T> : VboBase
        where T : struct
    {
        protected VboBase(int elementSize = -1, BufferTarget target = BufferTarget.ArrayBuffer)
            : base(elementSize < 0 ? 4 : elementSize, target)
        { }

        protected VboBase(int tCount, T[] initData = null, int elementSize = -1, BufferTarget target = BufferTarget.ArrayBuffer, BufferUsageHint hint = BufferUsageHint.DynamicDraw)
            : base(elementSize < 0 ? 4 : elementSize, target)
        {
            Init(tCount, initData, hint);
        }


        protected void Init(int tCount, T[] data = null, BufferUsageHint hint = BufferUsageHint.StaticRead)
        {
            Debug.Assert(tCount > 0, "Buffer count must be positive.");
            int tSize = Marshal.SizeOf(typeof(T));

            ByteCount = tCount * tSize;

            Bind();
            GL.BufferData(Target, ByteCount, data, hint);
        }

        protected void Update(T[] data, int tCount = -1, int offset = 0)
        {
            var tSize = Marshal.SizeOf(typeof(T));

            if (tCount == -1)
                tCount = Math.Min(ByteCount, data.Length * tSize);

            Bind();
            GL.BufferSubData(Target, new IntPtr(offset * tSize), tCount, data);
        }
    }

    public class Vbo<T> : VboBase<T>
        where T : struct
    {
        protected Vbo(int elementSize = -1, BufferTarget target = BufferTarget.ArrayBuffer)
            : base(elementSize, target)
        { }

        public Vbo(int tCount, T[] initData = null, int elementSize = -1, BufferTarget target = BufferTarget.ArrayBuffer, BufferUsageHint hint = BufferUsageHint.DynamicDraw)
            : base(tCount, initData, elementSize, target, hint)
        { }


        public new void Init(int tCount, T[] data = null, BufferUsageHint hint = BufferUsageHint.StaticRead)
        {
            base.Init(tCount, data, hint);
        }

        public new void Update(T[] data, int tCount = -1, int offset = 0)
        {
            base.Update(data, tCount, offset);
        }
    }

    internal sealed class StaticVbo<T> : VboBase<T>
        where T : struct
    {
        public StaticVbo(int tCount, T[] initData = null, int elementSize = -1, BufferTarget target = BufferTarget.ArrayBuffer, BufferUsageHint hint = BufferUsageHint.DynamicDraw)
            : base(tCount, initData, elementSize, target, hint)
        { }

        // These are shared by many geometries, prevent their disposal
        public override void Dispose()
        { }
    }
}
