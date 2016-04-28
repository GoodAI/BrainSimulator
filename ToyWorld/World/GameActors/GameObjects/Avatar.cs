﻿using System.Drawing;
﻿using System;
using System.Linq;
using VRageMath;
using World.GameActions;
using World.GameActors.Tiles;
using World.Physics;
using World.ToyWorldCore;

namespace World.GameActors.GameObjects
{
    public interface IAvatar : IAvatarControllable, ICharacter, IAutoupdateable, ICanPick
    {
        int Id { get; }
        IPickable Tool { get; set; }
    }

    public class Avatar : Character, IAvatar
    {
        public int Id { get; private set; }
        public IPickable Tool { get; set; }

        public int NextUpdateAfter { get; private set; }

        public float DesiredSpeed { get; set; }
        public float DesiredRotation { get; set; }
        public bool Interact { get; set; }
        public bool Use { get; set; }
        public bool PickUp { get; set; }
        public PointF Fof { get; set; }

        public Avatar(
            string tilesetName,
            int tileId,
            string name,
            int id,
            Vector2 initialPosition,
            Vector2 size,
            float direction = 0
            )
            : base(tilesetName, tileId, name, initialPosition, size, direction, typeof(CircleShape)
                )
        {
            Id = id;
            NextUpdateAfter = 1;
        }

        public void ResetControls()
        {
            DesiredSpeed = 0f;
            DesiredRotation = 0f;
            Interact = false;
            Use = false;
            PickUp = false;
            Fof = default(PointF);
        }

        public void AddToInventory(IPickable item)
        {
            if (Tool != null)
                return;

            Tool = item;
            Console.WriteLine("I picked up something.");
        }

        public void Update(IAtlas atlas)
        {
            if (PickUp)
            {
                GameActor target = atlas.ActorsInFrontOf(this, LayerType.Interactable).First();
                IInteractable interactableTarget = target as IInteractable;
                if (interactableTarget == null)
                    return;

                GameAction pickUpAction = new PickUp(this);
                interactableTarget.ApplyGameAction(atlas, pickUpAction);
            }
        }
    }
}
