using System;
using System.Drawing;

namespace GoodAI.ToyWorld.Control
{
    public enum RenderRequestImageCopyingMode
    {
        /// <summary>
        /// Copies data to the <see cref="ImageSettings.RenderedScene"/> array. Set to true if the cuda/opengl interop is failing.
        /// </summary>
        Cpu,
        /// <summary>
        /// Copies data to an OpenGL pixel buffer object and calls the supplied events with the buffer's handle.
        /// </summary>
        OpenglPbo,
    }

    /// <summary>
    /// 
    /// </summary>
    public class ImageSettings
    {
        /// <summary>
        /// Specifies what means should be used to gather the rendered scene.
        /// </summary>
        public RenderRequestImageCopyingMode CopyMode { get; set; }


        #region Cpu

        /// <summary>
        /// location where data is copied if it should be transfered through CPU
        /// </summary>
        public uint[] RenderedScene { get; private set; }

        #endregion

        #region Pbo

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

        #endregion
    }
}
