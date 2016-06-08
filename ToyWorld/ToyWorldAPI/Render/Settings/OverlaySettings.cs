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
        /// 
        /// </summary>
        public ToolBackgroundType ToolBackground { get; set; }

        #endregion


        public AvatarRROverlaySettings()
        {
            ToolBackground = ToolBackgroundType.BrownBorder;
        }
    }
}
