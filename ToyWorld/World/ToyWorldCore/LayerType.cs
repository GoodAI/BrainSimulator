using System;

namespace World.ToyWorldCore
{

    /// <summary>
    /// List of all Layers. Layers are loaded according to names; number determines order of rendering.
    /// We expect Avatar and Agent to be inside Object Layer.
    /// </summary>
    [Flags]
    public enum LayerType
    {
        Background = 1,
        OnBackground = 2,
        Path = 4,
        OnGroundInteractable = 8,
        ObstacleInteractable = 16,
        Obstacle = 32,
        Object = 64,
        Foreground = 128,
        ForegroundObject = 256,

        Obstacles = Obstacle | ObstacleInteractable,
        Interactable = OnGroundInteractable | ObstacleInteractable,
        All = Background | OnBackground | Path | OnGroundInteractable | ObstacleInteractable | Obstacle | Object | Foreground | ForegroundObject
    }
}
