using System;
using System.Collections.Generic;
using GoodAI.ToyWorld.Control;
using VRageMath;

namespace World.GameActors.GameObjects
{
    public interface IAvatar : IAvatarControlable, IDirection

    {
    }



    public class Avatar : Character, IAvatar
    {
        public readonly int Id;
        public sealed override string Name { get; protected set; }

        public float DesiredSpeed { get; set; }
        public float DesiredRotation { get; set; }
        public float ForwardSpeed { get; set; }
        public float RotationSpeed { get; set; }
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
            DesiredSpeed = 0f;
            DesiredRotation = 0f;
            Interact = false;
            Use = false;
            PickUp = false;
        }
    }
}
