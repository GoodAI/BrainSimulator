using System;
using System.Collections.Generic;
using GoodAI.ToyWorld.Control;
using World.GameActors.GameObjects;

namespace Game
{
    class AvatarController : IAvatarController
    {
        private Avatar m_avatar;
        private IAvatarPriorityActions a;

        public AvatarController(Avatar avatar)
        {
            m_avatar = avatar;
        }

        public void SetActions(IAvatarPriorityActions actions)
        {
            a.Copy(actions);
            m_avatar.Forward = actions.Forward.Value;
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
            m_avatar.ClearConstrols();
        }
    }
}
