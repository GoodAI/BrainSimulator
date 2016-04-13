using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using VRage.Collections;
using World.GameActors.Tiles;
using World.WorldInterfaces;

namespace World.ToyWorldCore
{
    public class AutoupdateRegister
    {
        private readonly CircularList<List<IAutoupdateable>> m_register;

        public int Size { get { return m_register.Size; } }

        public List<IAutoupdateable> CurrentUpdateRequests
        {
            get
            {
                Contract.Ensures(Contract.Result<List<IAutoupdateable>>() != null);
                return m_register[0];
            }
        }

        public AutoupdateRegister(int registerSize = 100)
        {
            Contract.Requires<ArgumentOutOfRangeException>(registerSize > 0, "Register size must be larger than zero.");

            m_register = new CircularList<List<IAutoupdateable>>(registerSize);
            m_register.MoveNext();
        }

        public void Register(IAutoupdateable actor, int timePeriod = 1)
        {
            Contract.Requires<ArgumentNullException>(actor != null, "You cannot register null object for updating.");
            Contract.Requires<ArgumentOutOfRangeException>(timePeriod > 0, "Update period has to be larger than zero.");

            m_register[timePeriod].Add(actor);
        }

        public void Tick()
        {
            m_register.MoveNext();
        }

        public void UpdateItems(IWorld world)
        {
            foreach (IAutoupdateable actor in CurrentUpdateRequests)
                actor.Update(world);
            CurrentUpdateRequests.Clear();
        }

        [ContractInvariantMethod]
        private void Invariants()
        {
            Contract.Invariant(m_register != null, "Register cannot be null");
        }
    }
}