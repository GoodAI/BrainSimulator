using System;
using System.Collections.Generic;
using GoodAI.ToyWorld.Control;

namespace World.GameActors.GameObjects
{
    public class Avatar : Character
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

        public void AddAction(AvatarAction<object> action)
        {
            if (AvatarActions.ContainsKey(action.ActionId))
            {
                int currentPriority = action.Priority;
                int prevPriority = AvatarActions[action.ActionId].Priority;
                if (currentPriority > prevPriority)
                {
                    AvatarActions[action.ActionId] = action;
                }
            }
            else
            {
                AvatarActions.Add(action.ActionId, action);
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
