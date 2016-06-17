using System;
using System.Drawing;

namespace GoodAI.ToyWorld.Control
{
    [Flags]
    public enum RenderRequestOverlay
    {
        None,

    }

    /// <summary>
    /// 
    /// </summary>
    public class OverlaySettings
    {
        /// <summary>
        /// Specifies which overlays should be used.
        /// </summary>
        public AvatarRenderRequestOverlay EnabledOverlays { get; set; }
    }


    [Flags]
    public enum AvatarRenderRequestOverlay
    {
        None,

        /// <summary>
        /// Draws the inventory tool to a corner of the screen
        /// </summary>
        InventoryTool,
    }

    public enum ToolBackgroundType
    {
        None,
        BrownBorder = 5,
        Brown = 6,
        GrayBorder = 9,
        Gray = 10,
    }

    public class AvatarRROverlaySettings : OverlaySettings
    {

        /// <summary>
        /// Specifies which overlays should be used.
        /// </summary>
        public new AvatarRenderRequestOverlay EnabledOverlays { get; set; }


        #region Inventory Tool

        /// <summary>
        /// The size of the inventory. Ranges from 0 to 2 (2 covers the whole screen).
        /// </summary>
        public PointF ToolSize { get; set; }

        /// <summary>
        /// The screen position of the tool overlay. Values are clamped to (-1,1).
        /// </summary>
        public PointF ToolPosition { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public ToolBackgroundType ToolBackground { get; set; }

        #endregion


        public AvatarRROverlaySettings()
        {
            const float toolSize = 0.08f;
            const float toolMargin = 0.05f;
            ToolSize = new PointF(toolSize, toolSize);
            ToolPosition = new PointF(1 - (toolMargin + toolSize * 0.5f), 1 - (toolMargin + toolSize * 0.5f));
            ToolBackground = ToolBackgroundType.BrownBorder;
        }
    }
}
