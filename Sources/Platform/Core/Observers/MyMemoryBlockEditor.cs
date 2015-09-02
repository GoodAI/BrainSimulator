using GoodAI.Core.Memory;
using GoodAI.Core.Utils;

namespace GoodAI.Core.Observers
{
    public class MyMemoryBlockEditor : MyMemoryBlockObserver
    {
        public void StartEdit()
        {
            Target.SafeCopyToHost();
        }

        public void EndEdit()
        {
            Target.SafeCopyToDevice();            
        }

        public void SetPixel<T>(int x, int y, T value) where T : struct
        {
            if (Target is MyMemoryBlock<T>)
            {
                int index = y * TextureWidth + x;
                T[] host = (Target as MyMemoryBlock<T>).Host;
                
                host[y * TextureWidth + x] = value;                                
            }
            else
            {
                MyLog.WARNING.WriteLine("Invalid data value");
            }
        }
    }
}
