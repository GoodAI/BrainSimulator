using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using VRage.Collections;
using World.GameActors;

namespace World.ToyWorldCore
{
    public class AutoupdateRegister
    {
        private CircularList<List<GameActor>> m_register;

        public AutoupdateRegister(int registerSize = 100)
        {
            Contract.Requires(registerSize > 0, "Register size must be larger than zero.");
            m_register = new CircularList<List<GameActor>>(registerSize);
        }

        public void Register(GameActor actor, int timePeriod = 1)
        {
            Contract.Requires<ArgumentNullException>(actor != null, "You cannot register null object for updating.");
            Contract.Requires<ArgumentOutOfRangeException>(timePeriod > 0, "Update period has to be larger than zero.");
            m_register[timePeriod].Add(actor);
        }

        //public List<Tile> NextUpdateRequests()
        //{
        //    throw new NotImplementedException();
        //}
    }
}