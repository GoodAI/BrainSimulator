using System;
using System.Collections.Generic;
using GoodAI.ToyWorld.Control;

namespace World.GameActors.GameObjects
{
    class Avatar : Character
    {
        public override string Name
        {
            get { throw new NotImplementedException(); }
        }

        internal List<IAvatarAction> AvatarActions
        {
            get
            {
                throw new System.NotImplementedException();
            }
            set
            {
            }
        }

        internal IUsable Tool
        {
            get
            {
                throw new System.NotImplementedException();
            }
            set
            {
            }
        }
    }
}
