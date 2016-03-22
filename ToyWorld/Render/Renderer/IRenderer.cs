using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Render;
using OpenTK;
using OpenTK.Graphics;

namespace Render.Renderer
{
    public interface IRenderer
    {
        INativeWindow Window { get; }
        IGraphicsContext Context { get; }

        // Allow only local creation of windows
        void CreateWindow(GraphicsMode graphicsMode, int width, int height);
        // Allow only local creation of contexts
        void CreateContext();

        void Init();
        void Reset();

        void EnqueueMessage(IRenderRequest request);
        void ProcessMessages(); // Each message is a render pass in general...
    }
}
