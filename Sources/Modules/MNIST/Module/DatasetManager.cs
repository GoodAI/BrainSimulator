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

        private List<int> _indexes;
        List<int>.Enumerator _indexer;

        List<int> _nExamplesPerClass;
    
        public DatasetManager(DatasetReaderFactory readerFactory)
        {
            _readerFactory = readerFactory;

            using (IDatasetReader r = _readerFactory.CreateReader())
            {
                int nClasses = r.NumClasses;
                _nExamplesPerClass = new List<int>(nClasses);
                _nExamplesPerClass.AddRange(Enumerable.Repeat(0, r.NumClasses));

                _examples = new List<IExample>();
                while (r.HasNext())
                {
                    IExample ex = r.ReadNext();
                    _examples.Add(ex);

                    _nExamplesPerClass[ex.Target]++;
                }
                //check if _examples.Count > 0?
            }

            int nIndexes = _examples.Count - 1;
            _indexes = new List<int>(nIndexes);
            _indexes.AddRange(Enumerable.Range(0, nIndexes));

            _indexer = _indexes.GetEnumerator();
        }

        public int GetMinMaxNumberPerClass()
        {
            return _nExamplesPerClass.Min();
        }

        public IExample GetNext()
        {
            if (_indexer.MoveNext() == false) {
                _indexer = _indexes.GetEnumerator();
                return GetNext();
            }

            return _examples[_indexer.Current];
        }
    }
}
