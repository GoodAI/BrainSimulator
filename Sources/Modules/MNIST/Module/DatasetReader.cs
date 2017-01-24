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

    public abstract class AbstractDatasetReaderFactory
    {
        public abstract IDatasetReader CreateReader();
    }
}
