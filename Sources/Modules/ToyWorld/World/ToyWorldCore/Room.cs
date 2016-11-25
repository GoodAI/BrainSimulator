namespace World.ToyWorldCore
{
    public class Room
    {
        public string Name { get; set; }

        public int Size { get; set; }

        public Room(string name)
        {
            Name = name;
        }
    }
}