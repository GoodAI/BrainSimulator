namespace GoodAI.ToyWorld.Control
{
    public enum AvatarActionEnum
    {
        Rotation,
        Acceleration,
        Use,
        Interact,
        Pick
    }

    public class AvatarAction<T>
    {
        private AvatarActionEnum m_actionId;
        private T m_value;
        private int m_priority;
    }
}
