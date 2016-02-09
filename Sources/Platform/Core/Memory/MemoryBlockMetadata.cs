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
        RenderingMethod
    }

    public interface IMemoryBlockMetadata : IDictionary<string, string>
    {
        string this[MemoryBlockMetadataKeys index] { get; set; }
        bool TryGetValue(MemoryBlockMetadataKeys key, out string value);
    }

    public class MemoryBlockMetadata : Dictionary<string, string>, IMemoryBlockMetadata
    {
        public string this[MemoryBlockMetadataKeys index]
        {
            get { return this[index.ToString()]; }
            set { this[index.ToString()] = value; }
        }

        public bool TryGetValue(MemoryBlockMetadataKeys key, out string value)
        {
            return TryGetValue(key.ToString(), out value);
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
