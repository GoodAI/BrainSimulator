using System;
using System.ComponentModel;

namespace MNIST
{
    public interface IDatasetReader : IDisposable
    {
        IExample ReadNext();
        bool HasNext();
        int NumClasses { get; }
    }

    public enum DatasetReaderFactoryType
    {
        Train, Test
    }

    public abstract class DatasetReaderFactory
    {
        private DatasetReaderFactoryType _type;

        public DatasetReaderFactory(DatasetReaderFactoryType type)
        {
            _type = type;
        }

        public IDatasetReader CreateReader()
        {
            switch (_type)
            {
                case DatasetReaderFactoryType.Train:
                    return CreateTrainReader();
                case DatasetReaderFactoryType.Test:
                    return CreateTestReader();
                default:
                    throw new InvalidEnumArgumentException();
            }
        }

        protected abstract IDatasetReader CreateTrainReader();
        protected abstract IDatasetReader CreateTestReader();
    }
}
