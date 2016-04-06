using System;
using System.Collections.Generic;
using GoodAI.ToyWorld.Control;
using VRageMath;

namespace World.GameActors.GameObjects
{
    public interface IAvatar : IControlable
    {
        Point Position { get; set; }
    }

    public class Avatar : Character, IAvatar
    {
        public readonly int Id;
        public sealed override string Name { get; protected set; }

        public float Acceleration { get; set; }
        public float Rotation { get; set; }
        public bool Interact { get; set; }
        public bool Use { get; set; }
        public bool PickUp { get; set; }

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

        public void ResetControls()
        {
            Acceleration = 0f;
            Rotation = 0;
            Interact = false;
            Use = false;
            PickUp = false;
        }
    }
}
