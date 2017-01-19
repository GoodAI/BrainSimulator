using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST
{
    public class Indexer
    {
        protected int[] _array;
        protected int _length;
        protected int _index;

        public Indexer(int[] array)
        {
            _array = array;
            _length = array.Length;
            _index = 0;
        }

        public void Resize(int length)
        {
            if (length < 1 || length > _array.Length)
            {
                throw new InvalidOperationException("Size of Indexer must fall within range [1, array.Length]");
            }

            Reset();
            _length = length; //TODO: get rid of side effect of ShuffleIndexer reshuffling itself before length is set
        }

        public int SampleRandom(Random random)
        {
            return _array[random.Next(_length)];
        }

        protected virtual void Reset()
        {
            _index = 0;
        }

        public int GetNext()
        {
            if (_index >= _length)
            {
                Reset();
            }

            return _array[_index++];
        }
    }

    public class ShuffleIndexer : Indexer
    {
        private Random _random;
        public ShuffleIndexer(int[] array, Random random) : base(array)
        {
            _random = random;
        }

        protected override void Reset()
        {
            base.Reset();

            for (int n = _length; n > 1; --n)
            {
                int i = _random.Next(n);
                int tmp = _array[i];
                _array[i] = _array[n - 1];
                _array[n - 1] = tmp;
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
        protected DatasetReaderFactory _readerFactory;
        protected List<IExample> _examples;

        protected ClassOrderOption _classOrder;
        protected ExampleOrderOption _exampleOrder;
        protected Random _random;

        protected Indexer[] _indexers;
        protected Indexer _classIndexer;

        protected int[] _nExamplesPerClass;
        protected int _nClasses;

        protected int[] _classFilter;
        protected bool _useClassFilter;

        protected bool _needLoad;

        public ClassOrderOption ClassOrder
        {
            get { return _classOrder; }
            set { _classOrder = value; }
        }

        public DatasetManager(DatasetReaderFactory readerFactory)
        {
            _readerFactory = readerFactory;
            _needLoad = true;
            _useClassFilter = false;
        }

        public void Init(int seed, ExampleOrderOption exampleOrder)
        {
            _random = seed == 0 ? new Random() : new Random(seed);
            _exampleOrder = exampleOrder;

            if (_needLoad)
            {
                LoadDataset();
                _needLoad = false;
            }

            Reindex();
        }

        private void LoadDataset()
        {
            Console.WriteLine("Dataset loading...");
            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();
            using (IDatasetReader r = _readerFactory.CreateReader())
            {
                _nClasses = r.NumClasses;
                _nExamplesPerClass = Enumerable.Repeat(0, _nClasses).ToArray();

                _examples = new List<IExample>();
                while (r.HasNext())
                {
                    IExample ex = r.ReadNext();
                    _examples.Add(ex);

                    _nExamplesPerClass[ex.Target]++;
                }
            }

            if (_examples.Count == 0)
            {
                throw new Exception("Dataset is empty!"); //TODO: better exception?
            }

            sw.Stop();
            Console.WriteLine("took: {0}", sw.Elapsed);
        }

        private void Reindex()
        {
            int[][] datasetIndices = new int[_nClasses][];

            for (int i = 0; i < _nClasses; ++i)
            {
                datasetIndices[i] = new int[_nExamplesPerClass[i]];
            }

            int[] idxs = Enumerable.Repeat(0, _nClasses).ToArray();
            for (int i = 0; i < _examples.Count; ++i)
            {
                int t = _examples[i].Target;
                datasetIndices[t][idxs[t]++] = i;
            }

            _indexers = new Indexer[_nClasses];
            for (int i = 0; i < _nClasses; ++i)
            {
                if (_exampleOrder == ExampleOrderOption.Shuffle)
                {
                    _indexers[i] = new ShuffleIndexer(datasetIndices[i], _random);
                }
                else
                {
                    _indexers[i] = new Indexer(datasetIndices[i]);
                }
            }
        }

        public void UseClassFilter(bool doUse)
        {
            if (_needLoad)
            {
                return;
            }

            _useClassFilter = doUse;

            if (_useClassFilter)
            {
                _classIndexer = new Indexer(_classFilter);
            }
            else
            {
                _classIndexer = new Indexer(Enumerable.Range(0, _nClasses).ToArray());
            }
        }

        public void SetClassFilter(string filter)
        {
            string[] strClasses = filter.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            // TODO: check all classes >= 0 && < _nClasses
            _classFilter = Array.ConvertAll(strClasses, int.Parse);
            Array.Sort(_classFilter);

            UseClassFilter(_useClassFilter);
        }

        public int SetExampleLimit(int limit)
        {
            if (_needLoad)
            {
                return limit;
            }

            limit = Math.Min(limit, _nExamplesPerClass.Max());

            for (int i = 0; i < _indexers.Length; ++i)
            {
                if (limit < _nExamplesPerClass[i])
                {
                    _indexers[i].Resize(limit);
                }
            }

            return limit;
        }

        public IExample GetNext()
        {
            int classNum;
            if (_classOrder == ClassOrderOption.Random)
            {
                classNum = _classIndexer.SampleRandom(_random);
            }
            else
            {
                classNum = _classIndexer.GetNext();
            }

            return GetNext(classNum);
        }

        public IExample GetNext(int classNum)
        {
            int idx;
            if (_exampleOrder == ExampleOrderOption.RandomSample)
            {
                idx = _indexers[classNum].SampleRandom(_random);
            }
            else
            {
                idx = _indexers[classNum].GetNext();
            }

            Console.WriteLine("Exaple id: {0}", idx);
            return _examples[idx];
        }
    }
}
