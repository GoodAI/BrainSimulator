using System;
using System.Collections.Generic;
using System.Drawing;
using System.Reflection;
using System.Windows.Forms;

namespace Common
{
    public static class Helpers
    {
        #region Conversions
            public static byte[] ToByte(Int16[] data)
            {
                if (data == null)
                    return null;

                byte[] data_out = new byte[data.Length * 2];
                for (int index = 0; index < data.Length; index++)
                {
                    Int16 sample = data[index];
                    data_out[index * 2] = (byte)Convert.ToByte(sample & 0xff);
                    data_out[(index * 2) + 1] = (byte)Convert.ToByte((sample >> 8) & 0xff);
                }

                return data_out;
            }

            public static byte[] ToByte(Int32[] data)
            {
                if (data == null)
                    return null;

                byte[] data_out = new byte[data.Length * 4];
                for (int index = 0; index < data.Length; index++)
                {
                    Int32 sample = data[index];
                    data_out[index * 2] = (byte)Convert.ToByte(sample & 0xff);
                    data_out[(index * 2) + 1] = (byte)Convert.ToByte((sample >> 8) & 0xff);
                }

                return data_out;
            }

            public static byte[] ToByte(Int64[] data)
            {
                if (data == null)
                    return null;

                byte[] data_out = new byte[data.Length * 8];
                for (int index = 0; index < data.Length; index++)
                {
                    Int64 sample = data[index];
                    data_out[index * 2] = (byte)Convert.ToByte(sample & 0xff);
                    data_out[(index * 2) + 1] = (byte)Convert.ToByte((sample >> 8) & 0xff);
                }

                return data_out;
            }

            public static byte[] ToByte(Single[] data)
            {
                if (data == null)
                    return null;
                
                List<byte> data_out = new List<byte>();
                for (int index = 0; index < data.Length; index++)
                {
                    Single sample = data[index];
                    byte[] temp = BitConverter.GetBytes(sample);
                    data_out.AddRange(temp);
                }

                return data_out.ToArray();
            }

            public static byte[] ToByte(double[] data)
            {
                if (data == null)
                    return null;

                List<byte> data_out = new List<byte>();
                for (int index = 0; index < data.Length; index++)
                {
                    double sample = data[index];
                    byte[] temp = BitConverter.GetBytes(sample);
                    data_out.AddRange(temp);
                }

                return data_out.ToArray();
            }


            public static Int16[] ToInt16(byte[] data)
            {
                if (data == null)
                    return null;

                int index = 0;
                Int16[] data_out = new Int16[data.Length / 2];
                for (int i = 0; i + 2 <= data.Length; i += 2)
                {
                    Int16 d = BitConverter.ToInt16(data, i);
                    data_out[index] = (Int16)(d == Int16.MinValue ? Int16.MinValue + 1 : d);
                    index++;
                }

                return data_out;
            }

            public static Int32[] ToInt32(byte[] data)
            {
                if (data == null)
                    return null;

                int index = 0;
                Int32[] data_out = new Int32[data.Length / 4];
                for (int i = 0; i + 4 <= data.Length; i += 4)
                {
                    Int32 d = BitConverter.ToInt32(data, i);
                    data_out[index] = (Int32)(d == Int32.MinValue ? Int32.MinValue + 1 : d);
                    index++;
                }

                return data_out;
            }
            public static Int32[] ToInt32(short[] data)
            {
                int[] res = new int[data.Length];
                for (int i = 0; i < data.Length; i++)
                    res[i] = (int)data[i];

                return res;
            }
            public static Int32[] ToInt32(float[] data)
            {
                int[] res = new int[data.Length];
                for (int i = 0; i < data.Length; i++)
                    res[i] = (int)data[i];

                return res;
            }
            public static Int32[] ToInt32(double[] data)
            {
                int[] res = new int[data.Length];
                for (int i = 0; i < data.Length; i++)
                    res[i] = (int)data[i];

                return res;
            }

            public static Int64[] ToInt64(byte[] data)
            {
                if (data == null)
                    return null;

                int index = 0;
                Int64[] data_out = new Int64[data.Length / 8];
                for (int i = 0; i + 8 <= data.Length; i += 8)
                {
                    Int64 d = BitConverter.ToInt64(data, i);
                    data_out[index] = (Int64)(d == Int64.MinValue ? Int64.MinValue + 1 : d);
                    index++;
                }

                return data_out;
            }

            public static Single[] ToFloat(byte[] data)
            {
                if (data == null)
                    return null;

                int index = 0;
                Single[] data_out = new Single[data.Length / 4];
                for (int i = 0; i + 4 <= data.Length; i += 4)
                {
                    Single d = BitConverter.ToSingle(data, i);
                    data_out[index] = (Single)(d == Single.MinValue ? Single.MinValue + 1 : d);
                    index++;
                }

                return data_out;
            }

            public static double[] ToDouble(byte[] data)
            {
                if (data == null)
                    return null;

                int index = 0;
                double[] data_out = new double[data.Length / 8];
                for (int i = 0; i + 8 <= data.Length; i += 8)
                {
                    double d = BitConverter.ToDouble(data, i);
                    data_out[index] = (double)(d == double.MinValue ? double.MinValue + 1 : d);
                    index++;
                }

                return data_out;
            }
        #endregion

        #region Maximum & Minimum searching
            #region Maximum
            public static byte Max(byte[] set) 
            {
                byte max = byte.MinValue;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] > max)
                        max = set[i];
                return max;
            }

            public static Int16 Max(Int16[] set)
            {
                Int16 max = Int16.MinValue;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] > max)
                        max = set[i];
                return max;
            }

            public static Int32 Max(Int32[] set)
            {
                Int32 max = Int32.MinValue;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] > max)
                        max = set[i];
                return max;
            }

            public static Int64 Max(Int64[] set)
            {
                Int64 max = Int64.MinValue;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] > max)
                        max = set[i];
                return max;
            }

            public static Single Max(Single[] set)
            {
                Single max = Single.MinValue;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] > max)
                        max = set[i];
                return max;
            }

            public static double Max(double[] set)
            {
                double max = double.MinValue;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] > max)
                        max = set[i];
                return max;
            }

            public static Point Max(Point[] set)
            {
                Point max = new Point(0,0);
                for (int i = 0; i < set.Length; i++)
                    if (set[i].Y > max.Y)
                        max = set[i];
                return max;
            }
            #endregion

            #region Absolute maximum
            public static byte AbsMax(byte[] set)
            {
                byte max = 0;
                for (int i = 0; i < set.Length; i++)
                    if (Math.Abs(set[i]) > Math.Abs(max))
                        max = set[i];
                return max;
            }

            public static Int16 AbsMax(Int16[] set)
            {
                Int16 max = 0;
                for (int i = 0; i < set.Length; i++)
                    if (Math.Abs(set[i]) > Math.Abs(max))
                        max = set[i];
                return max;
            }

            public static Int32 AbsMax(Int32[] set)
            {
                Int32 max = 0;
                for (int i = 0; i < set.Length; i++)
                    if (Math.Abs(set[i]) > Math.Abs(max))
                        max = set[i];
                return max;
            }

            public static Int64 AbsMax(Int64[] set)
            {
                Int64 max = 0;
                for (int i = 0; i < set.Length; i++)
                    if (Math.Abs(set[i]) > Math.Abs(max))
                        max = set[i];
                return max;
            }

            public static Single AbsMax(Single[] set)
            {
                Single max = 0;
                for (int i = 0; i < set.Length; i++)
                    if (Math.Abs(set[i]) > Math.Abs(max))
                        max = set[i];
                return max;
            }

            public static double AbsMax(double[] set)
            {
                double max = 0;
                for (int i = 0; i < set.Length; i++)
                    if (Math.Abs(set[i]) > Math.Abs(max))
                        max = set[i];
                return max;
            }

            public static Point AbsMax(Point[] set)
            {
                Point max = new Point(0,0);
                for (int i = 0; i < set.Length; i++)
                    if (Math.Abs(set[i].Y) > Math.Abs(max.Y))
                        max = set[i];
                return max;
            }
            #endregion

            #region Index of maximum
            public static long MaximumIndex(byte[] set)
            {
                byte max = byte.MinValue;
                long pos = 0;
                for (long i = 0; i < set.Length; i++)
                    if (set[i] > max)
                    {
                        max = set[i];
                        pos = i;
                    }
                return pos;
            }

            public static long MaximumIndex(Int16[] set)
            {
                Int16 max = Int16.MinValue;
                long pos = 0;
                for (long i = 0; i < set.Length; i++)
                    if (set[i] > max)
                    {
                        max = set[i];
                        pos = i;
                    }
                return pos;
            }

            public static long MaximumIndex(Int32[] set)
            {
                Int32 max = Int32.MinValue;
                long pos = 0;
                for (long i = 0; i < set.Length; i++)
                    if (set[i] > max)
                    {
                        max = set[i];
                        pos = i;
                    }
                return pos;
            }

            public static long MaximumIndex(Int64[] set)
            {
                Int64 max = Int64.MinValue;
                long pos = 0;
                for (long i = 0; i < set.Length; i++)
                    if (set[i] > max)
                    {
                        max = set[i];
                        pos = i;
                    }
                return pos;
            }

            public static long MaximumIndex(Single[] set)
            {
                Single max = Single.MinValue;
                long pos = 0;
                for (long i = 0; i < set.Length; i++)
                    if (set[i] > max)
                    {
                        max = set[i];
                        pos = i;
                    }
                return pos;
            }

            public static long MaximumIndex(double[] set)
            {
                double max = double.MinValue;
                long pos = 0;
                for (long i = 0; i < set.Length; i++)
                    if (set[i] > max)
                    {
                        max = set[i];
                        pos = i;
                    }
                return pos;
            }
            #endregion

            #region Minimum
            public static byte Min(byte [] set)
            {
                byte min = byte.MaxValue;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] < min)
                        min = set[i];
                return min;
            }

            public static Int16 Min(Int16[] set)
            {
                Int16 min = Int16.MaxValue;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] < min)
                        min = set[i];
                return min;
            }

            public static Int32 Min(Int32[] set)
            {
                Int32 min = Int32.MaxValue;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] < min)
                        min = set[i];
                return min;
            }

            public static Int64 Min(Int64[] set)
            {
                Int64 min = Int64.MaxValue;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] < min)
                        min = set[i];
                return min;
            }

            public static Single Min(Single[] set)
            {
                Single min = Single.MaxValue;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] < min)
                        min = set[i];
                return min;
            }

            public static double Min(double[] set)
            {
                double min = double.MaxValue;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] < min)
                        min = set[i];
                return min;
            }

            public static Point Min(Point[] set)
            {
                Point min = new Point(0,short.MaxValue);
                for (int i = 0; i < set.Length; i++)
                    if (set[i].Y < min.Y)
                        min = set[i];
                return min;
            }
            #endregion

            #region Asolute minimum
            public static byte AbsMin(byte[] set)
            {
                byte min = byte.MaxValue;
                for (int i = 0; i < set.Length; i++)
                    if (Math.Abs(set[i]) < Math.Abs(min))
                        min = set[i];
                return min;
            }

            public static Int16 AbsMin(Int16[] set)
            {
                Int16 min = Int16.MaxValue;
                for (int i = 0; i < set.Length; i++)
                    if (Math.Abs(set[i]) < Math.Abs(min))
                        min = set[i];
                return min;
            }

            public static Int32 AbsMin(Int32[] set)
            {
                Int32 min = Int32.MaxValue;
                for (int i = 0; i < set.Length; i++)
                    if (Math.Abs(set[i]) < Math.Abs(min))
                        min = set[i];
                return min;
            }

            public static Int64 AbsMin(Int64[] set)
            {
                Int64 min = Int64.MaxValue;
                for (int i = 0; i < set.Length; i++)
                    if (Math.Abs(set[i]) < Math.Abs(min))
                        min = set[i];
                return min;
            }

            public static Single AbsMin(Single[] set)
            {
                Single min = Single.MaxValue;
                for (int i = 0; i < set.Length; i++)
                    if (Math.Abs(set[i]) < Math.Abs(min))
                        min = set[i];
                return min;
            }

            public static double AbsMin(double[] set)
            {
                double min = double.MaxValue;
                for (int i = 0; i < set.Length; i++)
                    if (Math.Abs(set[i]) < Math.Abs(min))
                        min = set[i];
                return min;
            }

            public static Point AbsMin(Point[] set)
            {
                Point min = new Point(0,short.MaxValue);
                for (int i = 0; i < set.Length; i++)
                    if (Math.Abs(set[i].Y) < Math.Abs(min.Y))
                        min = set[i];
                return min;
            }
            #endregion

            #region Index of minimum
            public static long MinimumIndex(byte[] set)
            {
                byte min = byte.MaxValue;
                long pos = 0;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] < min)
                    {
                        min = set[i];
                        pos = i;
                    }
                return pos;
            }

            public static long MinimumIndex(Int16[] set)
            {
                Int16 min = Int16.MaxValue;
                long pos = 0;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] < min)
                    {
                        min = set[i];
                        pos = i;
                    }
                return pos;
            }

            public static long MinimumIndex(Int32[] set)
            {
                Int32 min = Int32.MaxValue;
                long pos = 0;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] < min)
                    {
                        min = set[i];
                        pos = i;
                    }
                return pos;
            }

            public static long MinimumIndex(Int64[] set)
            {
                Int64 min = Int64.MaxValue;
                long pos = 0;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] < min)
                    {
                        min = set[i];
                        pos = i;
                    }
                return pos;
            }

            public static long MinimumIndex(Single[] set)
            {
                Single min = Single.MaxValue;
                long pos = 0;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] < min)
                    {
                        min = set[i];
                        pos = i;
                    }
                return pos;
            }

            public static long MinimumIndex(double[] set)
            {
                double min = double.MaxValue;
                long pos = 0;
                for (int i = 0; i < set.Length; i++)
                    if (set[i] < min)
                    {
                        min = set[i];
                        pos = i;
                    }
                return pos;
            }
            #endregion
        #endregion

        #region Split channel
            public static void SplitChannels(byte[] wave, out byte[] _waveLeft, out byte[] _waveRight)
            {
                _waveLeft = new byte[wave.Length / 2];
                _waveRight = new byte[wave.Length / 2];

                // Split out channels from sample
                int h = 0;
                for (int i = 0; i < wave.Length; i += 4)
                {
                    Array.Copy(wave, i, _waveLeft, h, 2);
                    Array.Copy(wave, i + 2, _waveRight, h, 2);
                    h++;
                }
            }

            public static void SplitChannels(byte[] wave, out Int16[] _waveLeft, out Int16[] _waveRight)
            {
                _waveLeft = new Int16[wave.Length / 4];
                _waveRight = new Int16[wave.Length / 4];

                // Split out channels from sample
                int h = 0;
                for (int i = 0; i < wave.Length; i += 4)
                {
                    _waveLeft[h] = BitConverter.ToInt16(wave, i);
                    _waveRight[h] = BitConverter.ToInt16(wave, i + 2);
                    h++;
                }
            }

            public static void SplitChannels(Int16[] wave, out Int16[] _waveLeft, out Int16[] _waveRight)
            {
                _waveLeft = new Int16[wave.Length / 2];
                _waveRight = new Int16[wave.Length / 2];

                // Split out channels from sample
                int h = 0;
                for (int i = 0; i < wave.Length - 1; i += 2)
                {
                    _waveLeft[h] = wave[i];
                    _waveRight[h] = wave[i + 1];
                    h++;
                }
            }
        
        #endregion

        #region Combine channels
            public static byte[] CombineChannels(byte[] _waveLeft, byte[] _waveRight)
            {
                if (_waveLeft.Length != _waveRight.Length)
                    throw new Exception("Channels length not equal.");

                byte[] wave = new byte[_waveLeft.Length + _waveRight.Length];
                int h = 0;
                for (int i = 0; i < wave.Length; i += 4)
                {
                    Array.Copy(_waveLeft, h, wave, i, 2);
                    Array.Copy(_waveRight, h, wave, i + 2, 2);
                    h++;
                }
                return wave;
            }

            public static Int16[] CombineChannels(Int16[] _waveLeft, Int16[] _waveRight)
            {
                if (_waveLeft.Length != _waveRight.Length)
                    throw new Exception("Channels length not equal.");

                Int16[] wave = new Int16[_waveLeft.Length + _waveRight.Length];
                int h = 0;
                for (int i = 0; i < wave.Length; i += 2)
                {
                    Array.Copy(_waveLeft, h, wave, i, 1);
                    Array.Copy(_waveRight, h, wave, i + 1, 1);
                    h++;
                }
                return wave;
            }
        #endregion

    }//end class
}//end namespace
