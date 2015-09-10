using System.Collections;
using System.Collections.Generic;


namespace GoodAI.Core.Utils
{
    #region Bidirectional Hashtable
    public class BidirHashtable<TKey,TValue> : IDictionary<TKey, TValue>
    {
        #region Private Fields
        private Dictionary<TKey, TValue> m_Forward = new Dictionary<TKey, TValue>();
        private Dictionary<TValue, TKey> m_Backward = new Dictionary<TValue, TKey>();
        #endregion
 
        #region IDictionary<TKey,TValue> Members
        public void Add(TKey key, TValue value)
        {
                m_Forward.Add(key, value);
                m_Backward.Add(value, key);
        }
        public bool ContainsKey(TKey key)
        {
                return m_Forward.ContainsKey(key);
        }
        public ICollection<TKey> Keys
        {
                get { return m_Forward.Keys; }
        }
        public bool Remove(TKey key)
        {
                m_Backward.Remove(m_Forward[key]);
                return m_Forward.Remove(key);
        }
        public bool TryGetValue(TKey key, out TValue value)
        {
                return m_Forward.TryGetValue(key, out value);
        }
        public ICollection<TValue> Values
        {
                get { return m_Forward.Values; }
        }
        public TValue this[TKey key]
        {
                get { return m_Forward[key]; }
                set { m_Forward[key] = value; }
        }
        #endregion
 
        #region ICollection<KeyValuePair<TKey,TValue>> Members
        public void Add(KeyValuePair<TKey, TValue> item)
        {
                ((ICollection<KeyValuePair<TKey, TValue>>)m_Forward).Add(item);
                ((ICollection<KeyValuePair<TValue, TKey>>)m_Backward).Add(new KeyValuePair<TValue, TKey>(item.Value, item.Key));
        }
        public void Clear()
        {
                ((ICollection<KeyValuePair<TKey, TValue>>)m_Forward).Clear();
                ((ICollection<KeyValuePair<TValue, TKey>>)m_Backward).Clear();
        }
        public bool Contains(KeyValuePair<TKey, TValue> item)
        {
                return ((ICollection<KeyValuePair<TKey, TValue>>)m_Forward).Contains(item);
        }
        public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
        {
                ((ICollection<KeyValuePair<TKey, TValue>>)m_Forward).CopyTo(array, arrayIndex);
        }
        public int Count
        {
                get { return ((ICollection<KeyValuePair<TKey, TValue>>)m_Forward).Count; }
        }
        public bool IsReadOnly
        {
                get { return ((ICollection<KeyValuePair<TKey, TValue>>)m_Forward).IsReadOnly; }
        }
        public bool Remove(KeyValuePair<TKey, TValue> item)
        {
                m_Backward.Remove(item.Value);
                return ((ICollection<KeyValuePair<TKey, TValue>>)m_Forward).Remove(item);
        }
        #endregion
 
        #region IEnumerable<KeyValuePair<TKey,TValue>> Members
        public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
        {
                return ((IEnumerable<KeyValuePair<TKey, TValue>>)m_Forward).GetEnumerator();
        }
        #endregion
 
        #region IEnumerable Members
        IEnumerator IEnumerable.GetEnumerator()
        {
                return m_Forward.GetEnumerator();
        }
        #endregion
 
        #region Public Methods for Reverse Lookup
        public bool ContainsValue(TValue value)
        {
                return m_Backward.ContainsKey(value);
        }
        public TKey ReverseLookup(TValue value)
        {
                return m_Backward[value];
        }
        #endregion
      }
    #endregion
}