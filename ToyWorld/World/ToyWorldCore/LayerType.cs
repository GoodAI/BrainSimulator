namespace World.ToyWorldCore
{

    /// <summary>
    /// List of all Layers. Layers are loaded according to names; number determines order of rendering.
    /// We expect Avatar and Agent to be inside Object Layer.
    /// </summary>
    public enum LayerType
    {
        Background = 1,
        OnBackground = 2,
        Path = 3,
        OnGroundInteractable = 4,
        ObstacleInteractable = 5,
        Obstacle = 6,
        Object = 7,
        Foreground = 8,
        ForegroundObject = 9
    }
}
