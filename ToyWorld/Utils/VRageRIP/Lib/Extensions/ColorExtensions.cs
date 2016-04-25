namespace System
{
    public static class ColorExtensions
    {
        private static int[] _buffer = new int[0];


        public static int[] ArgbToRgbaArray(this IntPtr self, int pxCount)
        {
            if (_buffer.Length < pxCount)
                _buffer = new int[pxCount];

            unsafe
            {
                var ptr = (int*)self.ToPointer();

                for (int i = 0; i < pxCount; i++)
                {
                    int val = ptr[i];

                    var a = (byte)(val >> 24);
                    var r = (byte)(val >> 16);
                    var g = (byte)(val >> 8);
                    var b = (byte)val;


                    val |= a;
                    val <<= 8;
                    val |= r;
                    val <<= 8;
                    val |= g;
                    val <<= 8;
                    val |= b;

                    _buffer[i] = val;
                }
            }

            return _buffer;
        }
    }
}
