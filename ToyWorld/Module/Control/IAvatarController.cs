namespace GoodAI.ToyWorld.Control
{
    public interface IAvatarController
    {
        void SetAction<T>(AvatarAction<T> action);

        IStats GetStats();

        string GetComment();
    }
}
