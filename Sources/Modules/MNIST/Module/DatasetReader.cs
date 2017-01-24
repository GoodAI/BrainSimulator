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
        private DatasetReaderFactoryType m_type;

        public DatasetReaderFactory(DatasetReaderFactoryType type)
        {
            m_type = type;
        }

        public IDatasetReader CreateReader()
        {
            switch (m_type)
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
