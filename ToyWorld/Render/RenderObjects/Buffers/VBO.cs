using System;
using System.Runtime.InteropServices;
using OpenTK.Graphics.OpenGL;

namespace Render.RenderObjects.Buffers
{
    internal abstract class VboBase : IDisposable
    {
        public int Handle { get; private set; }

        public int ByteCount { get; private set; }

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

        protected void Init<T>(int tCount, T[] data, BufferUsageHint hint)
            where T : struct
        {
            int tSize = Marshal.SizeOf(typeof(T));

            Handle = GL.GenBuffer();
            ByteCount = tCount * tSize;

            Bind();
            GL.BufferData(Target, ByteCount, data, hint);
            //Unbind();
        }

        #endregion

        public virtual void Update<T>(T[] data, int tCount = -1, int offset = 0)
            where T : struct
        {
            var tSize = Marshal.SizeOf(typeof(T));

            if (tCount == -1)
                tCount = Math.Min(ByteCount, data.Length * tSize);

            Bind();
            GL.BufferSubData(Target, new IntPtr(offset * tSize), tCount, data);
            //Unbind();
        }

        public void Bind()
        {
            GL.BindBuffer(Target, Handle);
        }

        //public void Unbind()
        //{
        //    GL.BindBuffer(Target, 0);
        //}
    }

    internal class Vbo<T> : VboBase
        where T : struct
    {
        public Vbo(int tCount, T[] initData = null, int elementSize = -1, BufferTarget target = BufferTarget.ArrayBuffer, BufferUsageHint hint = BufferUsageHint.DynamicDraw)
            : base(elementSize < 0 ? 4 : elementSize, target)
        {
            Init(tCount, initData, hint);
        }
    }

    internal sealed class StaticVbo<T> : Vbo<T>
        where T : struct
    {
        public StaticVbo(int tCount, T[] initData = null, int elementSize = -1, BufferTarget target = BufferTarget.ArrayBuffer, BufferUsageHint hint = BufferUsageHint.DynamicDraw)
            : base(tCount, initData, elementSize, target, hint)
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
