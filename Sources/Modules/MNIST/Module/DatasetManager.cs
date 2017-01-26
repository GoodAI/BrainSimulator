using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using GoodAI.Core.Utils;
using System.Diagnostics;

namespace MNIST
{
    public class Indexer
    {
        private int[] m_array;
        private int m_length;
        private int m_index;

        private Random m_random;
        private bool m_doShuffle;

        public Indexer(int[] array, Random r, bool shuffleAtEnd = false, bool initShuffle = false)
        {
            m_array = array;
            m_length = array.Length;
            m_random = r;
            m_doShuffle = shuffleAtEnd;

            if (initShuffle)
            {
                Shuffle();
            }

            Reset();
        }

        private void Reset()
        {
            m_index = 0;
        }

        private void Shuffle()
        {
            for (int n = m_length; n > 1; --n)
            {
                int i = m_random.Next(n);
                int tmp = m_array[i];
                m_array[i] = m_array[n - 1];
                m_array[n - 1] = tmp;
            }
        }


        public void Resize(int length)
        {
            if (length < 1 || length > m_array.Length)
            {
                throw new ArgumentOutOfRangeException("Size of Indexer must fall within range <1, array.Length>");
            }

            m_length = length;
            Reset();
        }

        public int SampleRandom()
        {
            return m_array[m_random.Next(m_length)];
        }

        public int GetNext()
        {
            if (m_index >= m_length)
            {
                if (m_doShuffle)
                {
                    Shuffle();
                }

                Reset();
            }

            return m_array[m_index++];
        }
    }

    public enum ExampleOrderOption
    {
        NoShuffle,
        Shuffle,
        RandomSample,
    }

    public enum ClassOrderOption
    {
        Random,
        Increasing,
    }

    public class DatasetManager
    {
        private AbstractDatasetReaderFactory m_readerFactory;
        private List<IExample> m_examples;

        private ClassOrderOption m_classOrder;
        private ExampleOrderOption m_exampleOrder;
        private Random m_random;

        private Indexer[] m_indexers;
        private Indexer m_classIndexer;
        private int[] m_classFilter;

        private int[] m_nExamplesPerClass;
        private int m_exampleLimit;
        private int m_nClasses;

        private bool m_needLoad;
        private bool m_needInit;

        public ClassOrderOption ClassOrder
        {
            get { return m_classOrder; }
            set { m_classOrder = value; }
        }

        public DatasetManager(AbstractDatasetReaderFactory readerFactory)
        {
            m_readerFactory = readerFactory;
            m_needLoad = true;
            m_needInit = true;
        }

        public void Init(ExampleOrderOption exampleOrder, int seed = 0)
        {
            m_random = seed == 0 ? new Random() : new Random(seed);
            m_exampleOrder = exampleOrder;

            if (m_needLoad)
            {
                LoadDataset();
                m_needLoad = false;
            }

            InitExampleIndexers();
            InitClassIndexer();
            m_needInit = false;
        }

        private void LoadDataset()
        {
            Stopwatch stopWatch = new Stopwatch();

            stopWatch.Start();
            using (DatasetReader r = m_readerFactory.CreateReader())
            {
                m_nClasses = r.NumClasses;
                m_nExamplesPerClass = new int[m_nClasses];

                m_examples = new List<IExample>();
                while (r.HasNext())
                {
                    IExample ex = r.ReadNext();
                    m_examples.Add(ex);

                    m_nExamplesPerClass[ex.Target]++;
                }
            }
            stopWatch.Stop();

            m_exampleLimit = m_nExamplesPerClass.Max();
            MyLog.INFO.WriteLine("Loaded {0} examples in {1} s", m_examples.Count, stopWatch.ElapsedMilliseconds / 1000f);

            if (m_examples.Count == 0)
            {
                throw new FileLoadException("The loaded dataset is empty");
            }
        }

        private void InitExampleIndexers()
        {
            // for each class contains indices of m_examples with that class
            List<int>[] datasetIndices = new List<int>[m_nClasses];
            m_indexers = new Indexer[m_nClasses];

            for (int i = 0; i < m_nClasses; ++i)
            {
                datasetIndices[i] = new List<int>(m_nExamplesPerClass[i]);
            }

            for (int i = 0; i < m_examples.Count; ++i)
            {
                int t = m_examples[i].Target;
                datasetIndices[t].Add(i);
            }

            bool shuffleAtEnd = (m_exampleOrder == ExampleOrderOption.Shuffle);
            bool initShuffle = true;
            for (int i = 0; i < m_nClasses; ++i)
            {
                m_indexers[i] = new Indexer(datasetIndices[i].ToArray(), m_random, shuffleAtEnd, initShuffle);
            }
        }

        private void InitClassIndexer()
        {
            m_classFilter = Enumerable.Range(0, m_nClasses).ToArray();
            m_classIndexer = new Indexer(m_classFilter, m_random);
        }

        private void CheckWarnExampleLimit()
        {
            int max = int.MinValue;
            foreach (int classIdx in m_classFilter)
            {
                max = Math.Max(max, m_nExamplesPerClass[classIdx]);
            }

            if (m_exampleLimit > max)
            {
                MyLog.WARNING.WriteLine("Requested number of examples per class is higher than current highest number of examples per class ({0})", max);
            }
        }

        public void UseClassFilter(bool doUse, int[] filter = null)
        {
            if (m_needLoad)
            {
                return;
            }

            if (!doUse)
            {
                InitClassIndexer();
            }
            else
            {
                if (filter == null)
                {
                    throw new ArgumentNullException("When using filter, the filter must be provided");
                }

                m_classFilter = (int[]) filter.Clone();
                Array.Sort(m_classFilter);

                foreach (int c in m_classFilter)
                {
                    if (c < 0 || c >= m_nClasses)
                    {
                        throw new ArgumentOutOfRangeException(string.Format("Class filter must contain only numbers <0, {0}>", m_nClasses-1));
                    }
                }

                m_classIndexer = new Indexer(m_classFilter, m_random);
                CheckWarnExampleLimit();
            }
        }

        public int SetExampleLimit(int limit)
        {
            if (m_needLoad)
            {
                return limit;
            }

            m_exampleLimit = Math.Min(limit, m_nExamplesPerClass.Max());

            for (int i = 0; i < m_indexers.Length; ++i)
            {
                if (m_exampleLimit < m_nExamplesPerClass[i])
                {
                    m_indexers[i].Resize(m_exampleLimit);
                }
            }

            CheckWarnExampleLimit();
            return m_exampleLimit;
        }

        public IExample GetNext()
        {
            if (m_needInit)
            {
                throw new InvalidOperationException("DatasetManager is not initialized");
            }

            int classNum;

            if (m_classOrder == ClassOrderOption.Random)
            {
                classNum = m_classIndexer.SampleRandom();
            }
            else
            {
                classNum = m_classIndexer.GetNext();
            }

            return GetNext(classNum);
        }

        public IExample GetNext(int classNum)
        {
            if (m_needInit)
            {
                throw new InvalidOperationException("DatasetManager is not initialized");
            }

            int idx;

            if (m_exampleOrder == ExampleOrderOption.RandomSample)
            {
                idx = m_indexers[classNum].SampleRandom();
            }
            else
            {
                idx = m_indexers[classNum].GetNext();
            }

            return m_examples[idx];
        }
    }
}
