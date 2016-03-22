using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Render;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using Render.Renderer;
using Utils;

namespace Render
{
    public class GLRenderer : IRenderer
    {
        #region Fields

        private readonly Queue<IRenderRequest> m_renderRequestQueue = new Queue<IRenderRequest>();

        #endregion

        #region IRenderer overrides

        public INativeWindow Window { get; protected set; }
        public IGraphicsContext Context { get; protected set; }

        public void CreateWindow(GraphicsMode graphicsMode, int width, int height)
        {
            Window = new NativeWindow(width, height, Globals.AppName, GameWindowFlags.Default, graphicsMode, DisplayDevice.Default);
        }

        public void CreateContext()
        {
            Debug.Assert(Window != null, "Missing window, cannot create context.");

            if (Context == null)
                Context = new GraphicsContext(GraphicsMode.Default, Window.WindowInfo);

            Context.MakeCurrent(Window.WindowInfo);
            Context.LoadAll();
        }

        public void Init()
        {
            m_renderRequestQueue.Clear();

            GL.ClearColor(Color.Black);

        }

        public void Reset()
        {

            Init();
        }

        public void EnqueueMessage(IRenderRequest request)
        {
            Debug.Assert(request != null);
            m_renderRequestQueue.Enqueue(request);
        }

        public void ProcessMessages()
        {
            foreach (var renderRequest in m_renderRequestQueue)
            {
                Draw(renderRequest);
            }
        }

        #endregion


        void Draw(IRenderRequest request)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            // Context.SwapBuffers();
        }
    }
}
