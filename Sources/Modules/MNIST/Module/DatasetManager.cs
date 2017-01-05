using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST
{
    public class DatasetManager
    {
        private DatasetReaderFactory _readerFactory;
        private List<IExample> _examples;

        private int[] _indexes;
        int _indexer;

        private int[] _nExamplesPerClass;

        private Random _random;
        private bool _doShuffle;
        public DatasetManager(DatasetReaderFactory readerFactory, int seed, bool doShuffle)
        {
            _readerFactory = readerFactory;
            _doShuffle = doShuffle;
            _indexer = 0;

            if (seed == 0)
            {
                _random = new Random();
            }
            else
            {
                _random = new Random(seed);
            }

            //System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            //sw.Start();
            using (IDatasetReader r = _readerFactory.CreateReader())
            {
                int nClasses = r.NumClasses;
                _nExamplesPerClass = Enumerable.Repeat(0, r.NumClasses).ToArray();

                _examples = new List<IExample>();
                while (r.HasNext())
                {
                    IExample ex = r.ReadNext();
                    _examples.Add(ex);

                    _nExamplesPerClass[ex.Target]++;
                }
                //check if _examples.Count > 0?
            }

            _indexes = Enumerable.Range(0, _examples.Count - 1).ToArray();

            if (_examples.Count == 0)
            {
                throw new Exception("Dataset is empty!"); //TODO: better exception?
            }

            //sw.Stop();
            //Console.WriteLine("Elapsed={0}", sw.Elapsed);
        }

        public int GetMaxNumberPerClass()
        {
            return _nExamplesPerClass.Min();
        }
        
        private void CheckShuffle()
        {
            if (_doShuffle)
            {
                for (int n = _indexes.Length; n > 1; --n)
                {
                    int i = _random.Next(n);
                    int tmp = _indexes[i];
                    _indexes[i] = _indexes[n - 1];
                    _indexes[n - 1] = tmp;
                }
            }
        }

        public IExample GetNext()
        {

            if (_indexer == _indexes.Length) {
                _indexer = 0;
            }

            if (_indexer == 0)
            {
                CheckShuffle();
            }

            return _examples[_indexes[_indexer++]];
        }
    }
}
