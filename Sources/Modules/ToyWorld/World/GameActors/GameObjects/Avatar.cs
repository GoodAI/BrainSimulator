using System.Drawing;
using System.Linq;
using GoodAI.ToyWorldAPI;
using System.Collections.Generic;
using System;
﻿using System.Diagnostics;
﻿﻿using GoodAI.Logging;
using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActions;
using World.GameActors.Tiles.ObstacleInteractable;
﻿using World.Physics;

namespace World.GameActors.GameObjects
{
    public interface IAvatar : IAvatarControllable, ICharacter, IAutoupdateable, ICanPickGameObject, IMessageSender
    {
        /// <summary>
        /// Avatar unique Id for connecting from outside. Specified in .tmx file.
        /// </summary>
        int Id { get; }

        /// <summary>
        /// Energy of Avatar. In [0,1].
        /// </summary>
        float Energy { get; set; }

        /// <summary>
        /// Temperature of Avatar. In [0,2].
        /// </summary>
        float Temperature { get; set; }

        /// <summary>
        /// How much of rest Avatar had. Can be replenished only by sleeping.
        /// </summary>
        float Rested { get; set; }

        /// <summary>
        /// True if Avatar is controlled by somebody else
        /// </summary>
        bool PuppetControlled { get; set; }

        /// <summary>
        /// Tool in hand. Can be picked up and laid down.
        /// </summary>
        IPickableGameActor Tool { get; set; }
    }

    public class Avatar : Character, IAvatar
    {
        public event MessageEventHandler NewMessage = delegate { };

        private float m_energy;
        private float m_rested;
        private float m_temperature;
        private const float ENERGY_FOR_CARRYING = 0.001f;
        private const float ENERGY_FOR_EATING_FRUIT = 0.25f;
        private const float ENERGY_FOR_ACTION = 0.001f;
        private const float ENERGY_FOR_MOVE = 0.0001f;
        private const float ENERGY_FOR_ROTATION = 0.00001f;
        private const float ENERGY_FOR_LIVING = 0.00001f;
        private const float ENERGY_COEF_FOR_CATCHING_MOVING_OBJECT = 0.001f;
        private const float FATIGUE_FOR_LIVING = 0.000005f;
        private const int ENERGY_TO_HEAT_RATIO = 30;
        private const float TEMPERATURE_BALANCE_RATIO = 0.01f;
        public int Id { get; private set; }

        public float Energy
        {
            get { return m_energy; }
            set
            {
                m_energy = value;
                BoundValue(ref m_energy, 0, 1);
            }
        }

        public float Temperature
        {
            get { return m_temperature; }
            set
            {
                m_temperature = value;
                BoundValue(ref m_temperature, 0, 2);
            }
        }

        public float Rested
        {
            get { return m_rested; }
            set
            {
                m_rested = value;
                BoundValue(ref m_rested, 0, 1);
            }
        }

        public bool PuppetControlled { get; set; }

        public IPickableGameActor Tool { get; set; }

        public int NextUpdateAfter { get; private set; }

        public float DesiredSpeed { get; set; }
        public float DesiredLeftRotation { get; set; }
        public bool Interact { get; set; }
        public bool InteractPreviousStep { get; set; }
        public bool UseTool { get; set; }
        public bool UseToolPreviousStep { get; set; }
        public bool PickUp { get; set; }
        public bool PickUpPreviousStep { get; set; }
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
            Temperature = 1f;
            Rested = 1f;
            PuppetControlled = false;
            InitializeControls();
        }

        public void Update(IAtlas atlas)
        {
            var temperatureAround = atlas.Temperature(Position);

            //LogAvatarStatus(atlas, temperatureAround);

            float oldEnergy = Energy;
            LoseEnergy();
            BalanceTemperature(temperatureAround, oldEnergy - Energy);
            LoseRest();

            if (Interact)
            {
                if (!InteractPreviousStep)
                {
                    InteractWithAllInteractablesInFrontOf(atlas);
                }
                Interact = false;
                InteractPreviousStep = true;
                return;
            }
            InteractPreviousStep = false;

            if (PickUp)
            {
                if (!PickUpPreviousStep)
                {
                    if (Tool == null)
                    {
                    PerformPickup(atlas);
                    }
                    else
                    {
                    PerformLayDown(atlas);
                    }
                }
                PickUp = false;
                PickUpPreviousStep = true;
                return;
            }
            else
            {
                PickUpPreviousStep = false;
            }

            if (UseTool)
            {
                if (!UseToolPreviousStep)
                {
                    var usable = Tool as IUsableGameActor;

                    usable?.Use(new GameActorPosition(this, Position, LayerType.Object), atlas);
                }
                UseTool = false;
                UseToolPreviousStep = true;
            }
            else
            {
                UseToolPreviousStep = false;
            }
        }

        private void InteractWithAllInteractablesInFrontOf(IAtlas atlas)
        {
            List<GameActorPosition> tilesInFrontOf = GetInteractableTilesInFrontOf(atlas);
            foreach (GameActorPosition tileInFrontOf in tilesInFrontOf)
            {
                if (tileInFrontOf != null)
                {
                    var interactable = tileInFrontOf.Actor as IInteractableGameActor;


                    interactable?.ApplyGameAction(atlas, new Interact(this), tileInFrontOf.Position);
                    if (tileInFrontOf.Actor is Fruit)
                    {
                        EatFruit(tileInFrontOf);
                    }
                }
            }
        }

/*
        private void LogAvatarStatus(IAtlas atlas, float temperatureAround)
        {
            string areaName = atlas.AreasCarrier.AreaName(Position);

            if (areaName != null)
            {
                Log.Instance.Debug("Name of current Avatar's location is " + areaName + ".");
            }
            else
            {
                Log.Instance.Debug("Avatar is in unknown location.");
            }

            string roomName = atlas.AreasCarrier.RoomName(Position);

            if (roomName != null)
            {
                Log.Instance.Debug("Avatar is in room " + roomName + ".");
            }
            else
            {
                Log.Instance.Debug("Avatar is in no room.");
            }
            Log.Instance.Debug("Energy of avatar {" + Id + "} is " + Energy);

            Log.Instance.Debug("Temperature around avatar " + temperatureAround + ".");
            Log.Instance.Debug("Temperature of avatar " + Temperature + ".");
        }
*/

        private void BalanceTemperature(float temperatureAround, float energyDiff)
        {
            Temperature += energyDiff * ENERGY_TO_HEAT_RATIO;
            Temperature += (temperatureAround - Temperature) * TEMPERATURE_BALANCE_RATIO;
        }

        private void LoseRest()
        {
            Rested -= FATIGUE_FOR_LIVING;
        }

        private void InitializeControls()
        {
            ResetControls();
            InteractPreviousStep = false;
            UseToolPreviousStep = false;
            PickUpPreviousStep = false;
        }

        public void ResetControls()
        {
            DesiredSpeed = 0f;
            DesiredLeftRotation = 0f;
            Interact = false;
            UseTool = false;
            PickUp = false;
            Fof = default(PointF);
        }

        public bool AddToInventory(IPickableGameActor item)
        {
            if (Tool != null)
                return false;

            Tool = item;
            return true;
        }

        public IPickableGameActor RemoveFromInventory()
        {
            IPickableGameActor tool = Tool;
            Tool = null;
            return tool;
        }

        private void EatFruit(GameActorPosition applePosition)
        {
            var fruit = applePosition.Actor as Fruit;

            if (fruit is IEatable)
            {
                Energy += ENERGY_FOR_EATING_FRUIT;
            }
        }

        private void LoseEnergy()
        {
            if ((UseTool && !UseToolPreviousStep) || (Interact && !InteractPreviousStep) || (PickUp && !PickUpPreviousStep))
            {
                Energy -= ENERGY_FOR_ACTION;
            }

            Energy -= Math.Abs(DesiredSpeed) * ENERGY_FOR_MOVE;

            Energy -= Math.Abs(DesiredLeftRotation) * ENERGY_FOR_ROTATION;

            Energy -= ENERGY_FOR_LIVING;

            if (Tool is IGameObject)
            {
                Energy -= ENERGY_FOR_CARRYING * ((IGameObject)Tool).Weight;
            }
        }

        private bool PerformLayDown(IAtlas atlas)
        {
            Vector2 positionInFrontOf = atlas.PositionInFrontOf(this, 1);
            GameAction layDownAction = new LayDown(this);
            Tool.PickUp(atlas, layDownAction, positionInFrontOf);
            return true;
        }

        private bool PerformPickup(IAtlas atlas)
        {
            // check tile in front of
            List<GameActorPosition> targets = GetInteractableTilesInFrontOf(atlas);
            // if no tile, check objects
            if (targets.Count == 0)
            {
                targets = GetInteractableObjectsInFrontOf(atlas);
            }
            if (targets.Count == 0) return false;

            foreach (GameActorPosition target in targets)
            {
                IPickableGameActor interactableTarget = target.Actor as IPickableGameActor;
                if (interactableTarget == null) continue;
                GameAction pickUpAction = new PickUp(this);
                interactableTarget.PickUp(atlas, pickUpAction, target.Position);

                RemoveSpeed(target);
                return true;
            }
            return false;
        }

        private void RemoveSpeed(GameActorPosition target)
        {
            var actor = target.Actor as IForwardMovable;
            var gameObject = target.Actor as IGameObject;
            if (actor == null) return;
            Debug.Assert(gameObject != null, "gameObject != null");
            Energy -= actor.ForwardSpeed * gameObject.Weight * ENERGY_COEF_FOR_CATCHING_MOVING_OBJECT;
            actor.ForwardSpeed = 0;
        }

        private List<GameActorPosition> GetInteractableObjectsInFrontOf(IAtlas atlas)
        {
            // check circle in front of avatar
            float radius = ((CircleShape)PhysicalEntity.Shape).Radius;
            List<GameActorPosition> actorsInFrontOf = atlas.ActorsInFrontOf(this, LayerType.Object,
                0.2f + radius, radius).ToList();
            return actorsInFrontOf;
        }

        private List<GameActorPosition> GetInteractableTilesInFrontOf(IAtlas atlas)
        {
            List<GameActorPosition> actorsInFrontOf = atlas.ActorsInFrontOf(this, LayerType.Interactables).ToList();
            return actorsInFrontOf;
        }

        private void BoundValue(ref float value, float min, float max)
        {
            value = Math.Max(value, min);
            value = Math.Min(value, max);
        }
    }
}
