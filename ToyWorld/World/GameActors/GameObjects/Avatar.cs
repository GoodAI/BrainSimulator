﻿using System.Drawing;
using System.Linq;
using GoodAI.ToyWorldAPI;
﻿﻿using System.Collections.Generic;
﻿using System;
﻿﻿using System.Diagnostics;
﻿﻿using GoodAI.Logging;
using VRageMath;
using World.GameActions;
using World.GameActors.Tiles;
using World.Physics;
using World.ToyWorldCore;

namespace World.GameActors.GameObjects
{
    public interface IAvatar : IAvatarControllable, ICharacter, IAutoupdateable, ICanPick, IMessageSender
    {
        /// <summary>
        /// Avatar unique Id for connecting from outside. Specified in .tmx file.
        /// </summary>
        int Id { get; }

        /// <summary>
        /// Energy of Avatar. In [0,1].
        /// </summary>
        float Energy { get; }

        /// <summary>
        /// Tool in hand. Can be picked up and laid down.
        /// </summary>
        IPickable Tool { get; set; }
    }

    public class Avatar : Character, IAvatar
    {
        public event MessageEventHandler NewMessage = delegate { };

        private float m_energy;
        private const float ENERGY_FOR_CARRYING = 0.001f;
        private const float ENERGY_FOR_EATING_APPLE = 0.25f;
        private const float ENERGY_FOR_ACTION = 0.001f;
        private const float ENERGY_FOR_MOVE = 0.0001f;
        private const float ENERGY_FOR_ROTATION = 0.00001f;
        private const float ENERGY_FOR_LIVING = 0.00001f;
        private const float ENERGY_COEF_FOR_CATCHING_MOVING_OBJECT = 0.001f;
        public int Id { get; private set; }

        public float Energy
        {
            get { return m_energy; }
            private set
            {
                m_energy = value;
                BoundEnergy();
            }
        }

        public IPickable Tool { get; set; }

        public int NextUpdateAfter { get; private set; }

        public float DesiredSpeed { get; set; }
        public float DesiredRotation { get; set; }
        public bool Interact { get; set; }
        public bool UseTool { get; set; }
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
            Energy = 1f;
        }

        public void ResetControls()
        {
            DesiredSpeed = 0f;
            DesiredRotation = 0f;
            Interact = false;
            UseTool = false;
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
            Log.Instance.Debug("Energy of avatar {" +  Id + "} is " + Energy);
            LooseEnergy();

            if (Interact)
            {
                GameActorPosition tileInFrontOf = GetTilesInFrontOf(atlas);

                if (tileInFrontOf != null && (tileInFrontOf.Actor is Apple || tileInFrontOf.Actor is Pear))
                {
                    EatApple(atlas, tileInFrontOf);
                }

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

            if (UseTool)
            {
                
            }

        }

        private void EatApple(IAtlas atlas, GameActorPosition applePosition)
        {
            var apple = applePosition.Actor as Fruit;
            Debug.Assert(apple != null, "apple != null");
            apple.ApplyGameAction(atlas, new Interact(this), applePosition.Position);

            Energy += ENERGY_FOR_EATING_APPLE;
        }

        private void LooseEnergy()
        {
            if (UseTool || Interact || PickUp)
            {
                Energy -= ENERGY_FOR_ACTION;
            }

            Energy -= Math.Abs(DesiredSpeed)*ENERGY_FOR_MOVE;

            Energy -= Math.Abs(DesiredRotation)*ENERGY_FOR_ROTATION;

            Energy -= ENERGY_FOR_LIVING;

            if (Tool is IGameObject)
            {
                Energy -= ENERGY_FOR_CARRYING * ((IGameObject)Tool).Weight;
            }

            BoundEnergy();
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
                target = GetObjectInFrontOf(atlas);
            }
            if (target == null) return false;

            IInteractable interactableTarget = target.Actor as IInteractable;
            if (interactableTarget == null) return false;

            GameAction pickUpAction = new PickUp(this);
            interactableTarget.ApplyGameAction(atlas, pickUpAction, target.Position);

            RemoveSpeed(target);
            return true;
        }

        private void RemoveSpeed(GameActorPosition target)
        {
            var actor = target.Actor as IForwardMovable;
            var gameObject = target.Actor as ICharacter;
            if (actor != null)
            {
                Debug.Assert(gameObject != null, "gameObject != null");
                Energy -= actor.ForwardSpeed*gameObject.Weight*ENERGY_COEF_FOR_CATCHING_MOVING_OBJECT;
                actor.ForwardSpeed = 0;
            }
        }

        private GameActorPosition GetObjectInFrontOf(IAtlas atlas)
        {
            // check circle in front of avatar 
            float radius = ((CircleShape) PhysicalEntity.Shape).Radius;
            IEnumerable<GameActorPosition> actorsInFrontOf = atlas.ActorsInFrontOf(this, LayerType.Object,
                0.2f + radius, radius);
            GameActorPosition target = actorsInFrontOf.FirstOrDefault(x => x.Actor is IInteractable);
            return target;
        }

        private GameActorPosition GetTilesInFrontOf(IAtlas atlas)
        {
            GameActorPosition target = atlas.ActorsInFrontOf(this, LayerType.Interactable).FirstOrDefault();
            return target;
        }

        private void BoundEnergy()
        {
            if (m_energy > 1f)
            {
                m_energy = 1f;
            }
            if (m_energy < 0f)
            {
                m_energy = 0f;
            }
        }
    }
}
