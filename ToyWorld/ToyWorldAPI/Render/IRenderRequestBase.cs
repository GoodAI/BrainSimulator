using System;
using System.Drawing;

namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public interface IRenderRequestBase
    {
        /// <summary>
        /// 
        /// </summary>
        PointF PositionCenter { get; }

        /// <summary>
        /// 
        /// </summary>
        SizeF Size { get; set; }

        /// <summary>
        /// Determined by PositionCenter and Size.
        /// </summary>
        RectangleF View { get; }

        /// <summary>
        /// If true, renders the image upside down (flipped by the Y axis). The origin thus moves to the upper-left corner.
        /// </summary>
        bool FlipYAxis { get; set; }


        /// <summary>
        /// 
        /// </summary>
        Size Resolution { get; set; }

        /// <summary>
        /// The level of multisampling to use (each pixel uses 2^MultisampleLevel samples). Must be between 0 (no AA) and 5 (32x MSAA).
        /// </summary>
        int MultisampleLevel { get; set; }


        /// <summary>
        /// 
        /// </summary>
        bool GatherImage { get; set; }

        /// <summary>
        /// location where data is copied if it should be transfered through CPU
        /// </summary>
        uint[] Image { get; }

        /// <summary>
        /// set to true if the cuda/opengl interop is failing
        /// </summary>
        bool CopyImageThroughCpu { get; set; }

        /// <summary>
        /// Called before the timeframe in which the buffer object (VBO) will be used as a target to copy rendered results to.
        /// The argument is an OpenGL handle to the underlying buffer object.
        /// Use this callback to release any mapping related to the buffer object.
        /// This callback can be invoked from a different thread than the one calling MakeStep on GameController.
        /// </summary>
        event Action<IRenderRequestBase, uint> OnPreRenderingEvent;

        /// <summary>
        /// Called after the timeframe in which the buffer object (VBO) will be used as a target to copy rendered results to.
        /// The argument is an OpenGL handle to the underlying buffer object.
        /// Because an internal OpenGL context is now active, you can use this callback to do any copying
        /// from the buffer object or to map a CUDA pointer using CUDA-GL interop.
        /// This callback can be invoked from a different thread than the one calling MakeStep on GameController.
        /// </summary>
        event Action<IRenderRequestBase, uint> OnPostRenderingEvent;


        /// <summary>
        /// 
        /// </summary>
        bool DrawNoise { get; set; }

        /// <summary>
        /// 
        /// </summary>
        Color NoiseColor { get; set; }

        /// <summary>
        /// 
        /// </summary>
        float NoiseTransformationSpeedCoefficient { get; set; }

        /// <summary>
        /// 
        /// </summary>
        float NoiseMeanOffset { get; set; }
    }
}
