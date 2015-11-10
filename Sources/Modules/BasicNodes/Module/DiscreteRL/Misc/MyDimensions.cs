using GoodAI.Core.Utils;
using System.Collections.Generic;

namespace GoodAI.Modules.Harm
{

    /// <author>GoodAI</author>
    /// <meta>jv</meta>
    /// <status>Working</status>
    /// <summary>
    /// Implements multidimensional matrix as recursive list of lists of depth = no. dimensions.
    /// </summary>
    public interface IDimList
    {
        MyDimList GetChildNo(int ind);

        int Size();

        bool IsLast();

        float GetValue();

        void SetValue(float val);
    }

    public class MyDimList : IDimList
    {
        public static readonly float DEF_VAL = 0.0f;

        private List<MyDimList> m_myDims;
        private readonly int m_maxDepth;
        private readonly int m_depth;
        private float m_myVal;

        public MyDimList(int maxDepth, int depth)
        {
            this.m_maxDepth = maxDepth;
            this.m_depth = depth;
        }

        public MyDimList GetChildNo(int ind)
        {
            bool ll = this.IsLast();

            if (this.IsLast())
            {
                MyLog.ERROR.WriteLine("last dimension reached, I contain only numbers!");
                return null;
            }
            if (m_myDims == null)
            {
                m_myDims = new List<MyDimList>();
            }
            while (ind >= m_myDims.Count)
            {
                m_myDims.Add(new MyDimList(m_maxDepth, m_depth + 1));
            }
            if (ind < 0)
            {
                return m_myDims[0];
            }
            return m_myDims[ind];
        }

        public int Size()
        {
            return m_myDims.Count;
        }

        public bool IsLast()
        {
            return this.m_depth == this.m_maxDepth-1;
        }

        public float GetValue()
        {
            if (!this.IsLast())
            {
                MyLog.ERROR.WriteLine("This dimension is not last, will not get a value!");
                return DEF_VAL;
            }
            return m_myVal;
        }

        public void SetValue(float val)
        {
            if (!this.IsLast())
            {
                MyLog.ERROR.WriteLine("This dimension is not last, will not set a value!");
                return;
            }
            this.m_myVal = val;
        }
    }
}
