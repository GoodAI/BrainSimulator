using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST
{
    public class Indexer
    {
        protected int[] m_array;
        protected int m_length;
        protected int m_index;

        public Indexer(int[] array)
        {
            m_array = array;
            m_length = array.Length;
            m_index = 0;
        }

        public void Resize(int length)
        {
            if (length < 1 || length > m_array.Length)
            {
                throw new InvalidOperationException("Size of Indexer must fall within range [1, array.Length]");
            }

            Reset();
            m_length = length; //TODO: get rid of side effect of ShuffleIndexer reshuffling itself before length is set
        }

        public int SampleRandom(Random random)
        {
            return m_array[random.Next(m_length)];
        }

        protected virtual void Reset()
        {
            m_index = 0;
        }

        public int GetNext()
        {
            if (m_index >= m_length)
            {
                Reset();
            }

            return m_array[m_index++];
        }
    }

    public class ShuffleIndexer : Indexer
    {
        private Random m_random;
        public ShuffleIndexer(int[] array, Random random) : base(array)
        {
            m_random = random;
        }

        protected override void Reset()
        {
            base.Reset();

            for (int n = m_length; n > 1; --n)
            {
                int i = m_random.Next(n);
                int tmp = m_array[i];
                m_array[i] = m_array[n - 1];
                m_array[n - 1] = tmp;
            }
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
        protected DatasetReaderFactory m_readerFactory;
        protected List<IExample> m_examples;

        protected ClassOrderOption m_classOrder;
        protected ExampleOrderOption m_exampleOrder;
        protected Random m_random;

        protected Indexer[] m_indexers;
        protected Indexer m_classIndexer;

        protected int[] m_nExamplesPerClass;
        protected int m_nClasses;

        protected int[] m_classFilter;
        protected bool m_useClassFilter;

        protected bool m_needLoad;

        public ClassOrderOption ClassOrder
        {
            get { return m_classOrder; }
            set { m_classOrder = value; }
        }

        public DatasetManager(DatasetReaderFactory readerFactory)
        {
            m_readerFactory = readerFactory;
            m_needLoad = true;
            m_useClassFilter = false;
        }

        public void Init(int seed, ExampleOrderOption exampleOrder)
        {
            m_random = seed == 0 ? new Random() : new Random(seed);
            m_exampleOrder = exampleOrder;

            if (m_needLoad)
            {
                LoadDataset();
                m_needLoad = false;
            }

            Reindex();
        }

        private void LoadDataset()
        {
            Console.WriteLine("Dataset loading...");
            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();
            using (IDatasetReader r = m_readerFactory.CreateReader())
            {
                m_nClasses = r.NumClasses;
                m_nExamplesPerClass = Enumerable.Repeat(0, m_nClasses).ToArray();

                m_examples = new List<IExample>();
                while (r.HasNext())
                {
                    IExample ex = r.ReadNext();
                    m_examples.Add(ex);

                    m_nExamplesPerClass[ex.Target]++;
                }
            }

            if (m_examples.Count == 0)
            {
                throw new Exception("Dataset is empty!"); //TODO: better exception?
            }

            sw.Stop();
            Console.WriteLine("took: {0}", sw.Elapsed);
        }

        private void Reindex()
        {
            int[][] datasetIndices = new int[m_nClasses][];

            for (int i = 0; i < m_nClasses; ++i)
            {
                datasetIndices[i] = new int[m_nExamplesPerClass[i]];
            }

            int[] idxs = Enumerable.Repeat(0, m_nClasses).ToArray();
            for (int i = 0; i < m_examples.Count; ++i)
            {
                int t = m_examples[i].Target;
                datasetIndices[t][idxs[t]++] = i;
            }

            m_indexers = new Indexer[m_nClasses];
            for (int i = 0; i < m_nClasses; ++i)
            {
                if (m_exampleOrder == ExampleOrderOption.Shuffle)
                {
                    m_indexers[i] = new ShuffleIndexer(datasetIndices[i], m_random);
                }
                else
                {
                    m_indexers[i] = new Indexer(datasetIndices[i]);
                }
            }
        }

        public void UseClassFilter(bool doUse)
        {
            if (m_needLoad)
            {
                return;
            }

            m_useClassFilter = doUse;

            if (m_useClassFilter)
            {
                m_classIndexer = new Indexer(m_classFilter);
            }
            else
            {
                m_classIndexer = new Indexer(Enumerable.Range(0, m_nClasses).ToArray());
            }
        }

        public void SetClassFilter(string filter)
        {
            string[] strClasses = filter.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            // TODO: check all classes >= 0 && < m_nClasses
            m_classFilter = Array.ConvertAll(strClasses, int.Parse);
            Array.Sort(m_classFilter);

            UseClassFilter(m_useClassFilter);
        }

        public int SetExampleLimit(int limit)
        {
            if (m_needLoad)
            {
                return limit;
            }

            limit = Math.Min(limit, m_nExamplesPerClass.Max());

            for (int i = 0; i < m_indexers.Length; ++i)
            {
                if (limit < m_nExamplesPerClass[i])
                {
                    m_indexers[i].Resize(limit);
                }
            }

            return limit;
        }

        public IExample GetNext()
        {
            int classNum;
            if (m_classOrder == ClassOrderOption.Random)
            {
                classNum = m_classIndexer.SampleRandom(m_random);
            }
            else
            {
                classNum = m_classIndexer.GetNext();
            }

            return GetNext(classNum);
        }

        public IExample GetNext(int classNum)
        {
            int idx;
            if (m_exampleOrder == ExampleOrderOption.RandomSample)
            {
                idx = m_indexers[classNum].SampleRandom(m_random);
            }
            else
            {
                idx = m_indexers[classNum].GetNext();
            }

            Console.WriteLine("Exaple id: {0}", idx);
            return m_examples[idx];
        }
    }
}
