using System;
using System.Collections.Generic;
using GoodAI.ToyWorld.Control;

namespace World.GameActors.GameObjects
{
    public class Avatar : Character
    {
        public int Id { get; private set; }
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
            }
        }

        public Avatar(string name, int id)
        {
            Name = name;
            Id = id;
        }
    }
}
