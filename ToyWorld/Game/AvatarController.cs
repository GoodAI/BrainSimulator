using System;
using System.Collections.Generic;
using GoodAI.ToyWorld.Control;
using World.GameActors.GameObjects;

namespace Game
{
    class AvatarController : IAvatarController
    {
        private Avatar m_avatar;

        public AvatarController(Avatar avatar)
        {
            m_avatar = avatar;
        }

        public void SetAction(AvatarAction<object> action)
        {
            if (m_avatar.AvatarActions.ContainsKey(action.ActionId))
            {
                var currentPriority = action.Priority;
                var prevPriority = m_avatar.AvatarActions[action.ActionId].Priority;
                if (currentPriority > prevPriority)
                {
                    m_avatar.AvatarActions[action.ActionId] = action;
                }
            }
            else
            {
                m_avatar.AvatarActions.Add(action.ActionId, action);
            }
        }

        public IStats GetStats()
        {
            throw new NotImplementedException();
        }

        public string GetComment()
        {
            throw new NotImplementedException();
        }

        internal void ResetControls()
        {
            m_avatar.AvatarActions = new Dictionary<AvatarActionEnum, AvatarAction<object>>();
        }
    }
}
