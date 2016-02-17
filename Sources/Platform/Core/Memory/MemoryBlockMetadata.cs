using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenTK.Graphics;

namespace GoodAI.Core.Memory
{
    public enum MemoryBlockMetadataKeys
    {
        RenderingMethod,
        ShowCoordinates
    }

    public interface IMemoryBlockMetadata : IDictionary<string, object>
    {
        object this[MemoryBlockMetadataKeys index] { get; set; }
        bool TryGetValue(MemoryBlockMetadataKeys key, out object value);
        bool TryGetValue<T>(string key, out T value);
        bool TryGetValue<T>(MemoryBlockMetadataKeys key, out T value);
        T GetOrDefault<T>(string key, T defaultValue = default(T));
        T GetOrDefault<T>(MemoryBlockMetadataKeys key, T defaultValue = default(T));
    }

    public class MemoryBlockMetadata : Dictionary<string, object>, IMemoryBlockMetadata
    {
        public object this[MemoryBlockMetadataKeys index]
        {
            get { return this[index.ToString()]; }
            set { this[index.ToString()] = value; }
        }

        public bool TryGetValue(MemoryBlockMetadataKeys key, out object value)
        {
            return TryGetValue(key.ToString(), out value);
        }

        public bool TryGetValue<T>(string key, out T value)
        {
            object objectValue;
            if (base.TryGetValue(key, out objectValue))
            {
                try
                {
                    value = (T) objectValue;
                    return true;
                }
                catch
                {
                    // Default is used (see below).
                }
            }

            value = default(T);
            return false;
        }

        public bool TryGetValue<T>(MemoryBlockMetadataKeys key, out T value)
        {
            return TryGetValue(key.ToString(), out value);
        }

        public T GetOrDefault<T>(string key, T defaultValue = default(T))
        {
            T value;
            if (TryGetValue(key, out value))
                return value;

            return defaultValue;
        }

        public T GetOrDefault<T>(MemoryBlockMetadataKeys key, T defaultValue = default(T))
        {
            return GetOrDefault(key.ToString(), defaultValue);
        }
    }

    public enum RenderingMethod
    {
        RedGreenScale,
        GrayScale,
        ColorScale,
        BlackWhite,
        Vector,
        Raw,
        RGB
    }
}
