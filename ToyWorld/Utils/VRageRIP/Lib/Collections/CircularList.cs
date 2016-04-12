using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Diagnostics.Contracts;

namespace VRage.Collections
{
    /// <summary>
    /// Implements circular list. All indexing is done with respect to the current cursor position - i.e. the cursor is invisible to clients.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    [ContractVerification(true)]
    public class CircularList<T> : IEnumerator<T>, IEnumerable<T> where T : new()
    {
        protected readonly T[] m_circle;
        protected int m_cursor;

        [ExcludeFromCodeCoverage]
        object IEnumerator.Current { get { return Current; } }

        public T Current
        {
            get
            {
                RangeOK(m_cursor);
                Contract.Assume(m_cursor >= 0);
                return m_circle[m_cursor];
            }
        }

        public int Size { get { return m_circle.Length; } }

        public CircularList(int size)
        {
            Contract.Requires(size > 0, "Circular list needs size > 0");

            m_circle = new T[size];
            for (int i = 0; i < m_circle.Length; ++i)
                m_circle[i] = new T();

            m_cursor = -1;
        }

        public T this[int index]
        {
            get
            {
                Contract.Ensures(Contract.Result<T>() != null);
                Contract.Assume((m_cursor + index) % m_circle.Length >= 0);
                int accessIndex = (m_cursor + index) % m_circle.Length;
                return m_circle[accessIndex];
            }
            set
            {
                Contract.Assume((m_cursor + index) % m_circle.Length >= 0);
                int accessIndex = (m_cursor + index) % m_circle.Length;
                m_circle[accessIndex] = value;
            }
        }

        private void RangeOK(int index)
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
            if (++m_cursor >= m_circle.Length)
                m_cursor = 0;
            return true;
        }

        public IEnumerator<T> GetEnumerator()
        {
            return this;
        }

        [ExcludeFromCodeCoverage]
        IEnumerator IEnumerable.GetEnumerator()
        {
            return this;
        }

        public void Dispose() { }

        [ContractInvariantMethod]
        private void Invariants()
        {
            Contract.Invariant(m_circle != null);
            Contract.Invariant(m_circle.Length > 0);
            Contract.Invariant(m_cursor < m_circle.Length);
        }
    }
}
