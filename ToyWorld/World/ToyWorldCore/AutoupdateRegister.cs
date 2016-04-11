using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using World.GameActors;

namespace World.ToyWorldCore
{
    /// <summary>
    /// Implements circular list. All indexing is done with respect to the current cursor position - i.e. the cursor is invisible to clients.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class CircularList<T> : IEnumerator<T>, IEnumerable<T>
    {
        private T[] m_circle;
        private int m_cursor;

        object IEnumerator.Current { get { return Current; } }

        public T Current
        {
            get
            {
                RangeOK(m_cursor);
                return m_circle[m_cursor];
            }
        }

        public CircularList(int size)
        {
            Contract.Requires(size > 0, "Circular list needs size > 0");
            m_circle = new T[size];
            m_cursor = -1;
        }

        public T this[int index]
        {
            get
            {
                int accessIndex = m_cursor + index;
                RangeOK(accessIndex);
                return m_circle[accessIndex];
            }
        }

        public void RangeOK(int index)
        {
            if (index < 0)
                throw new ArgumentOutOfRangeException("Trying to access the element before the list start.");

            if (index >= m_circle.Length)
                throw new ArgumentOutOfRangeException("Trying to access the element after the list end.");
        }

        public void Reset()
        {
            m_cursor = -1;
        }

        public bool MoveNext()
        {
            if (++m_cursor > m_circle.Length)
                m_cursor = 0;
            return true;
        }

        public IEnumerator<T> GetEnumerator()
        {
            return this;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this;
        }

        public void Dispose() { }
    }

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

        }

        //public List<Tile> NextUpdateRequests()
        //{
        //    throw new NotImplementedException();
        //}
    }
}