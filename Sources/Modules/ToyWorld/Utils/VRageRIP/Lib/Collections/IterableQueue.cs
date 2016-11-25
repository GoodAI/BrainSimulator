using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using VRage.Collections;

namespace VRage.Collections
{
    public class IterableQueue<T> : MyQueue<T>, IEnumerable<T>
    {
        public IterableQueue()
            : base(200)
        { }

        public IterableQueue(int capacity)
            : base(capacity)
        { }

        public IterableQueue(IEnumerable<T> collection)
            : base(collection)
        { }


        #region IEnumerable<> overrides

        public IEnumerator<T> GetEnumerator()
        {
            if (m_head == m_tail)
                yield break;

            if (m_head < m_tail)
            {
                for (int i = m_head; i < m_tail; i++)
                    yield return m_array[i];
            }
            else
            {
                for (int i = m_head; i < m_array.Length; i++)
                    yield return m_array[i];

                for (int i = 0; i < m_tail; i++)
                    yield return m_array[i];
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        #endregion
    }
}
