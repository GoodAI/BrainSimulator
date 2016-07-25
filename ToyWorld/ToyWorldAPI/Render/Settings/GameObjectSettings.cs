using System;
using System.Drawing;

namespace GoodAI.ToyWorld.Control
{
    [Flags]
    public enum RenderRequestGameObject
    {
        None,

        /// <summary>
        /// Specifies, if visible tile layers should be drawn.
        /// </summary>
        TileLayers = 1,
        /// <summary>
        /// Specifies whether visible object layers should be drawn.
        /// </summary>
        ObjectLayers = 1 << 1,
    }

    /// <summary>
    /// 
    /// </summary>
    public struct GameObjectSettings
    {
        /// <summary>
        /// Specifies which effects should be used.
        /// </summary>
        public RenderRequestGameObject EnabledEffects { get; set; }


        /// <summary>
        /// If set, the world will use perspective instead of orthogonal projection (the world will appear three-dimensional).
        /// </summary>
        public bool Use3D { get; set; }


        #region Tile settings

        #endregion


        public GameObjectSettings(RenderRequestGameObject enabledEffects)
            : this()
        {
            EnabledEffects = enabledEffects;
        }
    }
}
