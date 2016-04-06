using System;
using System.Collections.Generic;
using GoodAI.ToyWorld.Control;
using VRageMath;

namespace World.GameActors.GameObjects
{
    public interface IAvatar
    {
        string Name { get; }
        float Acceleration { get; set; }
        float Rotation { get; set; }
        bool Interact { get; set; }
        bool Use { get; set; }
        bool PickUp { get; set; }
        Point Position { get; set; }
        void ClearConstrols();
    }

    public class Avatar : Character, IControlable, IAvatar
    {
        public readonly int Id;
        public sealed override string Name { get; protected set; }

        public float Acceleration { get; set; }
        public float Rotation { get; set; }
        public bool Interact { get; set; }
        public bool Use { get; set; }
        public bool PickUp { get; set; }

        private Dictionary<AvatarActionEnum, AvatarAction<object>> AvatarActions { get; set; }

        internal IUsable Tool
        {
            get
            {
                throw new NotImplementedException();
            }
            set
            {
                throw new NotImplementedException();
            }
        }

        public Avatar(string name, int id)
        {
            Name = name;
            Id = id;
        }

        public void ClearConstrols()
        {
            AvatarActions = new Dictionary<AvatarActionEnum, AvatarAction<object>>();
        }
    }
}
