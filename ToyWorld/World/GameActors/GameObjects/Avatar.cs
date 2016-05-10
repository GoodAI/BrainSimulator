﻿using System.Drawing;
using System.Linq;
using GoodAI.ToyWorldAPI;
﻿﻿using System.Collections.Generic;
using VRageMath;
using World.GameActions;
using World.GameActors.Tiles;
using World.Physics;
using World.ToyWorldCore;

namespace World.GameActors.GameObjects
{
    public interface IAvatar : IAvatarControllable, ICharacter, IAutoupdateable, ICanPick, IMessageSender
    {
        int Id { get; }
        IPickable Tool { get; set; }
    }

    public class Avatar : Character, IAvatar
    {
        public event MessageEventHandler NewMessage = delegate { };

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

        public bool AddToInventory(IPickable item)
        {
            if (Tool != null)
                return false;

            Tool = item;
            return true;
        }

        public IPickable RemoveFromInventory()
        {
            IPickable tool = Tool;
            Tool = null;
            return tool;
        }

        public void Update(IAtlas atlas)
        {
            if (Interact)
            {

                Interact = false;
                return;
            }

            if (PickUp)
            {
                if (Tool == null)
                {
                    PerformPickup(atlas);
                }
                else
                {
                    PerformLayDown(atlas);
                }
                PickUp = false;
                return;
            }

            if (Use)
            {
                
            }
        }

        private bool PerformLayDown(IAtlas atlas)
        {
            var positionInFrontOf = atlas.PositionInFrontOf(this, 1);
            GameAction layDownAction = new LayDown(this);
            Tool.ApplyGameAction(atlas, layDownAction, positionInFrontOf);
            return true;
        }

        private bool PerformPickup(IAtlas atlas)
        {
            // check tile in front of
            var target = GetTilesInFrontOf(atlas);
            // if no tile, check objects
            if (target == null)
            {
                target = GetObjectsInFrontOf(atlas, target);
            }
            if (target == null) return false;

            IInteractable interactableTarget = target.Actor as IInteractable;
            if (interactableTarget == null) return false;

            GameAction pickUpAction = new PickUp(this);
            interactableTarget.ApplyGameAction(atlas, pickUpAction, target.Position);

            ResetSpeed(target);
            return true;
        }

        private static void ResetSpeed(GameActorPosition target)
        {
            var actor = target.Actor as IForwardMovable;
            if (actor != null)
            {
                actor.ForwardSpeed = 0;
            }
        }

        private GameActorPosition GetObjectsInFrontOf(IAtlas atlas, GameActorPosition target)
        {
            // check circle in front of avatar 
            float radius = ((CircleShape) PhysicalEntity.Shape).Radius;
            IEnumerable<GameActorPosition> actorsInFrontOf = atlas.ActorsInFrontOf(this, LayerType.Object,
                0.2f + radius, radius);
            target = actorsInFrontOf.FirstOrDefault(x => x.Actor is IInteractable);
            return target;
        }

        private GameActorPosition GetTilesInFrontOf(IAtlas atlas)
        {
            GameActorPosition target = atlas.ActorsInFrontOf(this, LayerType.Interactable).FirstOrDefault();
            return target;
        }
    }
}
