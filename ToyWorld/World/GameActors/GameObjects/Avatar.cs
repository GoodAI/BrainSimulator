using System;
using System.Collections.Generic;
using GoodAI.ToyWorld.Control;

namespace World.GameActors.GameObjects
{
    public class Avatar : Character, IAvatarActions
    {
        public readonly int Id;
        public sealed override string Name { get; protected set; }

        public Dictionary<AvatarActionEnum, AvatarAction<object>> AvatarActions { get; set; }

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

        public float Forward { get; set; }
        public float Back { get; set; }
        public bool DoPickup { get; set; }
    }
}
