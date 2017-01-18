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

        public void Resize(int n)
        {
            _length = n;

            if (_length < 1 || _length > _array.Length)
            {
                throw new InvalidOperationException("Size of Indexer must fall within range [1, array.Length]");
            }
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
            if (_index == _length)
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


        public DatasetManager(DatasetReaderFactory readerFactory, ExampleOrderOption exampleOrder, int seed)
        {
            _readerFactory = readerFactory;
            _exampleOrder = exampleOrder;

            _random = seed == 0 ? new Random() : new Random(seed);

            //System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            //sw.Start();
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

            CreateIndexers();
            //sw.Stop();
            //Console.WriteLine("Elapsed={0}", sw.Elapsed);
        }

        private void CreateIndexers()
        {
            int[][] dsIndices = new int[_nClasses][];
            int[] indices2 = Enumerable.Repeat(0, _nClasses).ToArray(); //read as indices^2

            for (int i = 0; i < _nClasses; ++i)
            {
                dsIndices[i] = new int[_nExamplesPerClass[i]];
            }

            for (int i = 0; i < _examples.Count; ++i)
            {
                int t = _examples[i].Target;
                dsIndices[t][indices2[t]++] = i;
            }

            _indexers = new Indexer[_nClasses];
            for (int i = 0; i < _nClasses; ++i)
            {
                if (_exampleOrder == ExampleOrderOption.Shuffle)
                {
                    _indexers[i] = new ShuffleIndexer(dsIndices[i], _random);
                }
                else
                {
                    _indexers[i] = new Indexer(dsIndices[i]);
                }
            }
        }

        public int GetMaxNumberPerClass()
        {
            return _nExamplesPerClass.Min();
        }

        public void SetClassOrder(ClassOrderOption classOrder, string filter = null)
        {
            int[] classes;
            _classOrder = classOrder;

            if (filter == null)
            {
                classes = Enumerable.Range(0, _nClasses).ToArray();
            }
            else
            {
                string[] strClasses = filter.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                // TODO: check all classes >= 0 && < _nClasses
                classes = Array.ConvertAll(strClasses, int.Parse);
            }

            if (_classOrder == ClassOrderOption.Increasing)
            {
                Array.Sort(classes);
            }

            _classIndexer = new Indexer(classes);
        }

        public IExample GetNext()
        {
            if (_classOrder == ClassOrderOption.Random)
            {
                return GetNext(_classIndexer.SampleRandom(_random));
            }
            else
            {
                return GetNext(_classIndexer.GetNext());
            }
        }

        public IExample GetNext(int classNum)
        {
            if (_exampleOrder == ExampleOrderOption.RandomSample)
            {
                return _examples[_indexers[classNum].SampleRandom(_random)];
            }
            else
            {
                return _examples[_indexers[classNum].GetNext()];
            }
        }
    }
}
