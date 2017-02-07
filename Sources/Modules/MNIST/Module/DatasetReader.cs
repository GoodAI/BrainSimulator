using System;
using GoodAI.Core.Utils;

namespace MNIST
{
    public abstract class DatasetReader : IDisposable
    {
        public abstract IExample ReadNext();
        public abstract bool HasNext();
        public abstract void Dispose();

        public abstract int NumClasses { get; }
        protected static string BaseDir => MyResources.GetMyAssemblyPath() + @"\res\";
    }

    public abstract class AbstractDatasetReaderFactory
    {
        public abstract DatasetReader CreateReader();
    }
}
