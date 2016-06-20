using System;

namespace GoodAI.ToyWorld.Control
{
    public enum RenderRequestImageCopyingMode
    {
        /// <summary>
        /// No copying anywhere (scene is stored in internal framebuffer).
        /// </summary>
        None,
        /// <summary>
        /// Data will be copied to default window's framebuffer (accessible through the world's renderer object).
        /// </summary>
        DefaultFbo,
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
    public struct ImageSettings
    {
        /// <summary>
        /// Specifies what means should be used to gather the rendered scene.
        /// </summary>
        public RenderRequestImageCopyingMode CopyMode { get; set; }


        #region Cpu

        /// <summary>
        /// Called after a scene image was copied to the buffer.
        /// </summary>
        public event Action<IRenderRequestBase, uint[]> OnSceneBufferPrepared;

        /// <summary>
        /// You should not use this method. Stay away!
        /// </summary>
        public void InvokePostBufferPrepared(IRenderRequestBase renderRequest, uint[] buffer)
        {
            var callback = OnSceneBufferPrepared;

            if (callback != null)
                callback(renderRequest, buffer);
        }

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
        /// You should not use this method. Stay away!
        /// </summary>
        public void InvokePreRenderingEvent(IRenderRequestBase renderRequest, uint pboHandle)
        {
            var preCopyCallback = OnPreRenderingEvent;

            if (preCopyCallback != null)
                preCopyCallback(renderRequest, pboHandle);
        }

        /// <summary>
        /// Called after the timeframe in which the buffer object (VBO) will be used as a target to copy rendered results to.
        /// The argument is an OpenGL handle to the underlying buffer object.
        /// Because an internal OpenGL context is now active, you can use this callback to do any copying
        /// from the buffer object or to map a CUDA pointer using CUDA-GL interop.
        /// This callback can be invoked from a different thread than the one calling MakeStep on GameController.
        /// </summary>
        public event Action<IRenderRequestBase, uint> OnPostRenderingEvent;

        /// <summary>
        /// You should not use this method. Stay away!
        /// </summary>
        public void InvokePostRenderingEvent(IRenderRequestBase renderRequest, uint pboHandle)
        {
            var postCopyCallback = OnPostRenderingEvent;

            if (postCopyCallback != null)
                postCopyCallback(renderRequest, pboHandle);
        }

        #endregion


        public ImageSettings(RenderRequestImageCopyingMode copyingMode)
            : this()
        {
            CopyMode = copyingMode;
        }
    }
}
