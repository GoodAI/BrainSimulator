using System;
using System.Drawing;

namespace GoodAI.ToyWorld.Control
{
    public enum RenderRequestMultisampleLevel
    {
        None,
        x4 = 2,
        x8,
        x16,
    }

    /// <summary>
    /// 
    /// </summary>
    public interface IRenderRequestBase
        : IDisposable
    {
        /// <summary>
        /// Use this method to remove the request from processing queue. Dispose works too.
        /// </summary>
        void UnregisterRenderRequest();


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
        /// The level of multisampling to use (each pixel uses 2^MultisampleLevel samples). Must be between 0 (no AA) and 4 (16x MSAA).
        /// Currently level 1 is equal to level 2 (both are 4x MSAA).
        /// </summary>
        RenderRequestMultisampleLevel MultisampleLevel { get; set; }


        /// <summary>
        /// Contains the desired settings of rendered image copying.
        /// </summary>
        ImageSettings Image { get; set; }


        /// <summary>
        /// Contains the desired settings of scene effects.
        /// </summary>
        EffectSettings Effects { get; set; }


        /// <summary>
        /// Contains the desired settings for scene postprocessing.
        /// </summary>
        PostprocessingSettings Postprocessing { get; set; }


        /// <summary>
        /// Contains the desired settings for scene overlays.
        /// </summary>
        OverlaySettings Overlay { get; set; }
    }
}
