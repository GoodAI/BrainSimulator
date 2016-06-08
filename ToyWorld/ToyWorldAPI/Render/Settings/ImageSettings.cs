using System;
using System.Drawing;

namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public class ImageSettings
    {
        /// <summary>
        /// 
        /// </summary>
        public bool GatherImage { get; set; }

        /// <summary>
        /// set to true if the cuda/opengl interop is failing
        /// </summary>
        public bool CopyImageThroughCpu { get; set; }

        /// <summary>
        /// location where data is copied if it should be transfered through CPU
        /// </summary>
        public uint[] RenderedScene { get; private set; }

        /// <summary>
        /// Called before the timeframe in which the buffer object (VBO) will be used as a target to copy rendered results to.
        /// The argument is an OpenGL handle to the underlying buffer object.
        /// Use this callback to release any mapping related to the buffer object.
        /// This callback can be invoked from a different thread than the one calling MakeStep on GameController.
        /// </summary>
        public event Action<IRenderRequestBase, uint> OnPreRenderingEvent;

        /// <summary>
        /// Called after the timeframe in which the buffer object (VBO) will be used as a target to copy rendered results to.
        /// The argument is an OpenGL handle to the underlying buffer object.
        /// Because an internal OpenGL context is now active, you can use this callback to do any copying
        /// from the buffer object or to map a CUDA pointer using CUDA-GL interop.
        /// This callback can be invoked from a different thread than the one calling MakeStep on GameController.
        /// </summary>
        public event Action<IRenderRequestBase, uint> OnPostRenderingEvent;


        public ImageSettings()
        { }
    }
}
