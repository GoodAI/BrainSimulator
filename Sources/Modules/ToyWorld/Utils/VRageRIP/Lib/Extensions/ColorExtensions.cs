namespace System
{
    public static class ColorExtensions
    {
        private static int[] _buffer = new int[0];


        public static int[] ArgbToArgbArray(this IntPtr self, int pxCount)
        {
            if (_buffer.Length < pxCount)
                _buffer = new int[pxCount];

            unsafe
            {
                var ptr = (int*)self.ToPointer();

                for (int i = 0; i < pxCount; i++)
                {
                    int val = ptr[i];

                    _buffer[i] = val;
                }
            }

            return _buffer;
        }
    }
}
