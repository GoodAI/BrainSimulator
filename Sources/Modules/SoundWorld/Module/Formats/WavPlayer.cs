using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace GoodAI.Modules.SoundProcessing
{
    // Event delegates
    public delegate void BytePlaybackEventHandler(byte[] data);
    public delegate void ShortPlaybackEventHandler(short[] data);
    public delegate void PositionChangeEventHandler(long position);
    public delegate void PlaybackFinnishEventHandler();

    public class WavPlayer : IDisposable
    {
        //const
        public int BUFF_SIZE = 4096;
        public const byte BUFF_CNT = 3;

        // flags
        public bool HasTranscription = false;

        #region Internal classes
        /// <summary>
        /// Internal class for storing interval data from transcription file
        /// </summary>
        internal class Intervals
        {
            public char Feature { get; set; }
            public int Start { get; set; }
            public int Stop { get; set; }

            public Intervals(string feature, int start, int stop)
            {
                this.Feature = feature[0];
                this.Start = start;
                this.Stop = stop;
            }
        }

        /// <summary>
        /// Internal class for translation of exception messages
        /// </summary>
        internal class WaveOutHelper
        {
            public static void Try(int err)
            {
                if (err != WaveNative.MMSYSERR_NOERROR)
                    throw new Exception(err.ToString());
            }
        }//end class

        /// <summary>
        /// Internal class for Buffer list
        /// </summary>
        internal class WaveBuffer : IDisposable
        {
            #region Declaration
            public WaveBuffer NextBuffer;

            private AutoResetEvent m_PlayEvent = new AutoResetEvent(false);
            private IntPtr m_Wave;

            private byte[] m_HeaderData;
            private WaveNative.WaveHdr m_Header;
            private GCHandle m_HeaderHandle;
            private GCHandle m_HeaderDataHandle;

            private bool m_Playing;
            #endregion

            // Properties
            public int Size
            {
                get { return m_Header.dwBufferLength; }
            }

            public IntPtr Data
            {
                get { return m_Header.lpData; }
            }

            [STAThread]
            internal static void ProcessMsg(IntPtr hdrvr, int uMsg, int dwUser, ref WaveNative.WaveHdr wavhdr, int dwParam2)
            {
                switch (uMsg)
                {
                    case WaveNative.MM_WOM_OPEN:
                        break;
                    case WaveNative.MM_WOM_DONE:
                        try
                        {
                            GCHandle h = (GCHandle)wavhdr.dwUser;
                            WaveBuffer buf = (WaveBuffer)h.Target;
                            buf.OnCompleted();
                        }
                        catch { }
                        break;
                    case WaveNative.MM_WOM_CLOSE:
                        break;
                }
            }

            public WaveBuffer(IntPtr waveOutHandle, int size)
            {
                m_Wave = waveOutHandle;

                m_HeaderHandle = GCHandle.Alloc(m_Header, GCHandleType.Pinned);
                m_Header.dwUser = (IntPtr)GCHandle.Alloc(this);
                m_HeaderData = new byte[size];
                m_HeaderDataHandle = GCHandle.Alloc(m_HeaderData, GCHandleType.Pinned);
                m_Header.lpData = m_HeaderDataHandle.AddrOfPinnedObject();
                m_Header.dwBufferLength = size;
                WaveNative.waveOutPrepareHeader(m_Wave, ref m_Header, Marshal.SizeOf(m_Header));
            }

            public bool Write()
            {
                lock (this)
                {
                    m_PlayEvent.Reset();
                    m_Playing = WaveNative.waveOutWrite(m_Wave, ref m_Header, Marshal.SizeOf(m_Header)) == WaveNative.MMSYSERR_NOERROR;
                    return m_Playing;
                }
            }

            public void WaitFor()
            {
                if (m_Playing)
                {
                    m_Playing = m_PlayEvent.WaitOne();
                }
                else
                {
                    Thread.Sleep(0);
                }
            }

            public void OnCompleted()
            {
                m_PlayEvent.Set();
                m_Playing = false;
            }

            public void Dispose()
            {
                if (m_Header.lpData != IntPtr.Zero)
                {
                    WaveNative.waveOutUnprepareHeader(m_Wave, ref m_Header, Marshal.SizeOf(m_Header));
                    m_HeaderHandle.Free();
                    m_Header.lpData = IntPtr.Zero;
                }
                m_PlayEvent.Close();
                if (m_HeaderDataHandle.IsAllocated)
                    m_HeaderDataHandle.Free();
                GC.SuppressFinalize(this);
            }

            ~WaveBuffer()
            {
                Dispose();
            }

        }//end class

        #endregion

        #region Declarations

        public Stream m_stream;
        public WaveFormat m_format;

        public int m_length;
        public long m_position;
        private long m_start_pos;

        // flags
        public bool is_playing = false;
        private byte m_zero;

        //var
        private Thread m_Thread;
        private IntPtr m_WavePtr;
        private WaveBuffer m_Buffers;                       // linked list
        private WaveBuffer m_CurrentBuffer;

        private Intervals[] m_intervals;

        #endregion

        // callbacks
        private static WaveNative.WaveDelegate m_BufferProc = new WaveNative.WaveDelegate(WaveBuffer.ProcessMsg);

        #region Events
        public event PositionChangeEventHandler PositionChange;
        public event BytePlaybackEventHandler BytePlayback;
        public event ShortPlaybackEventHandler ShortPlayback;
        public event PlaybackFinnishEventHandler PlaybackFinnish;

        protected virtual void OnPositionChange(long position)
        {
            if (PositionChange != null)
                PositionChange(position);
        }

        protected virtual void OnBytePlayback(byte[] data)
        {
            if (BytePlayback != null)
                BytePlayback(data);
        }

        protected virtual void OnShortPlayback(short[] data)
        {
            if (ShortPlayback != null)
                ShortPlayback(data);
        }

        protected virtual void OnPlaybackFinnish()
        {
            if (PlaybackFinnish != null)
                PlaybackFinnish();
        }
        #endregion

        /// <summary>
        /// Low-level .wav format audio player.
        /// </summary>
        /// <param name="filename">Path to audio file to play.</param>
        public WavPlayer(string filename) : this(filename, -1, 4096) { }

        /// <summary>
        /// Low-level .wav format audio player.
        /// </summary>
        /// <param name="filename">Path to audio file to play.</param>
        /// <param name="device_id">Id of system device to play (default value is -1).</param>
        /// <param name="buff_size">Size of each batch to be sent to the device.</param>
        public WavPlayer(string filename, int device_id, int buff_size)
        {
            if (!File.Exists(filename))
                throw new Exception("File not found!");

            try
            {
                BUFF_SIZE = buff_size;
                // Open file, read header and open playback device
                m_stream = new FileStream(filename, FileMode.Open, FileAccess.Read);
                ReadHeader();
                m_start_pos = m_stream.Position;

                m_zero = m_format.wBitsPerSample == 8 ? (byte)128 : (byte)0;
                WaveOutHelper.Try(WaveNative.waveOutOpen(out m_WavePtr, device_id, m_format, m_BufferProc, 0, WaveNative.CALLBACK_FUNCTION));
                AllocateBuffers(BUFF_SIZE, BUFF_CNT);
            }
            catch 
            {
                m_stream.Close();
            }
        }

        /// <summary>
        /// Low-level .wav format audio player.
        /// </summary>
        /// <param name="stream">Audio stream to play.</param>
        public WavPlayer(Stream stream, WaveFormat format) : this(stream, format, -1, 4096) { }

        /// <summary>
        /// Low-level .wav format audio player.
        /// </summary>
        /// <param name="stream">Audio stream to play.</param>
        /// <param name="format">Audio format of stream to be played.</param>
        /// <param name="device_id">Id of system device to play (default value is -1).</param>
        /// <param name="buff_size">Size of each batch to be sent to the device.</param>
        public WavPlayer(Stream stream, WaveFormat format, int device_id, int buff_size)
        {
            if (stream == null)
                throw new Exception("Invalid stream!");

            try
            {
                // Set the buf size, open stream, prepare format and open playback device
                BUFF_SIZE = buff_size;
                m_stream = stream;
                m_format = format;
                m_start_pos = m_stream.Position;
                m_length = (int)(stream.Length / 2);

                // Try to read header if there's any
                BinaryReader Reader = new BinaryReader(m_stream);
                if (ReadChunk(Reader) == "RIFF")
                {
                    m_stream.Position = m_start_pos;
                    ReadHeader();
                }

                // Open WaveOut
                m_zero = m_format.wBitsPerSample == 8 ? (byte)128 : (byte)0;
                WaveOutHelper.Try(WaveNative.waveOutOpen(out m_WavePtr, device_id, m_format, m_BufferProc, 0, WaveNative.CALLBACK_FUNCTION));
                AllocateBuffers(BUFF_SIZE, BUFF_CNT);
            }
            catch (Exception e) { throw new Exception(e.Message); }
        }

        #region Playback functions
        /// <summary>
        /// Start playback
        /// </summary>
        public void Play()
        {
            if (is_playing)
                return;
            else
                is_playing = true;

            m_Thread = new Thread(new ThreadStart(ThreadPlay));
            m_Thread.Start();
        }

        /// <summary>
        /// Start playback from stream.
        /// </summary>
        /// <param name="stream">Source stream.</param>
        public void Play(Stream stream)
        {
            if (stream == null)
                throw new Exception("Invalid stream!");

            m_stream = stream;
            if (is_playing)
                return;
            else
                is_playing = true;

            m_Thread = new Thread(new ThreadStart(ThreadPlay));
            m_Thread.Start();
        }

        /// <summary>
        /// Temporarily pause playback
        /// </summary>
        public void Pause()
        {
            is_playing = false;

            if (m_Thread != null)
            {
                m_Thread.Abort();
                m_Thread.Join();
            }
        }

        /// <summary>
        /// Stop playback
        /// </summary>
        public void Stop()
        {
            m_stream.Close();
            is_playing = false;

            if (m_stream.CanRead)
                m_stream.Position = m_start_pos;

            if (m_Thread != null)
            {
                m_Thread.Abort();
                m_Thread.Join();
            }
        }

        /// <summary>
        /// Get list of available playback devices.
        /// </summary>
        /// <returns>List of names.</returns>
        public string[] GetDevicesList()
        {
            List<string> DevList = new List<string>();

            // get number of available output devices
            int waveOutDevicesCount = WaveNative.waveOutGetNumDevs();
            if (waveOutDevicesCount > 0)
            {
                for (int uDeviceID = 0; uDeviceID < waveOutDevicesCount; uDeviceID++)
                {
                    WaveNative.WaveOutCaps waveOutCaps = new WaveNative.WaveOutCaps();
                    WaveNative.waveOutGetDevCaps(uDeviceID, ref waveOutCaps, Marshal.SizeOf(typeof(WaveNative.WaveOutCaps)));
                    DevList.Add(new string(waveOutCaps.szPname).Remove(
                                new string(waveOutCaps.szPname).IndexOf('\0')).Trim());
                }
            }
            return DevList.ToArray();
        }

        private void ThreadPlay()
        {
            while (m_stream.Position != m_stream.Length)
            {
                // playback
                Advance();
                byte[] m_PlayBuffer = new byte[BUFF_SIZE];
                m_stream.Read(m_PlayBuffer, 0, BUFF_SIZE);
                Marshal.Copy(m_PlayBuffer, 0, m_CurrentBuffer.Data, m_PlayBuffer.Length);
                m_CurrentBuffer.Write();

                // Position change event
                OnPositionChange(m_stream.Position);

                // access low level playback data
                OnBytePlayback(m_PlayBuffer);
                short[] data = new short[m_PlayBuffer.Length / 2];
                Buffer.BlockCopy(m_PlayBuffer, 0, data, 0, m_PlayBuffer.Length);
                OnShortPlayback(data);
            }

            is_playing = false;
            m_stream.Position = m_start_pos;
            OnPlaybackFinnish();
        }

        private void Advance()
        {
            m_CurrentBuffer = m_CurrentBuffer == null ? m_Buffers : m_CurrentBuffer.NextBuffer;
            m_CurrentBuffer.WaitFor();
        }

        private void AllocateBuffers(int bufferSize, int bufferCount)
        {
            FreeBuffers();
            if (bufferCount > 0)
            {
                m_Buffers = new WaveBuffer(m_WavePtr, bufferSize);
                WaveBuffer Prev = m_Buffers;
                try
                {
                    for (int i = 1; i < bufferCount; i++)
                    {
                        WaveBuffer Buf = new WaveBuffer(m_WavePtr, bufferSize);
                        Prev.NextBuffer = Buf;
                        Prev = Buf;
                    }
                }
                finally
                {
                    Prev.NextBuffer = m_Buffers;
                }
            }
        }

        private void FreeBuffers()
        {
            m_CurrentBuffer = null;
            if (m_Buffers != null)
            {
                WaveBuffer First = m_Buffers;
                m_Buffers = null;

                WaveBuffer Current = First;
                do
                {
                    WaveBuffer Next = Current.NextBuffer;
                    Current.Dispose();
                    Current = Next;
                } while (Current != First);
            }
        }
        #endregion

        
        private string ReadChunk(BinaryReader reader)
        {
            byte[] ch = new byte[4];
            reader.Read(ch, 0, ch.Length);
            return Encoding.ASCII.GetString(ch);
        }

        /// <summary>
        /// Read RIFF header from current stream
        /// </summary>
        private void ReadHeader()
        {
            BinaryReader Reader = new BinaryReader(m_stream);
            if (ReadChunk(Reader) != "RIFF")
                throw new Exception("Invalid file format - not a RIFF!");

            Reader.ReadInt32();                                                 // File length minus first 8 bytes of RIFF description, we don't use it

            if (ReadChunk(Reader) != "WAVE")
                throw new Exception("Invalid file format - not a WAVE!");

            if (ReadChunk(Reader) != "fmt ")
                throw new Exception("Invalid file format - not an fmt!");

            int len = Reader.ReadInt32();
            if (len < 16)                                                       // bad format chunk length
                throw new Exception("Invalid file format - bad chunk length!");

            m_format = new WaveFormat(22050, 16, 2);                            // initialize to any format and fill it with valid info
            m_format.wFormatTag = Reader.ReadInt16();
            m_format.nChannels = Reader.ReadInt16();
            m_format.nSamplesPerSec = Reader.ReadInt32();
            m_format.nAvgBytesPerSec = Reader.ReadInt32();
            m_format.nBlockAlign = Reader.ReadInt16();
            m_format.wBitsPerSample = Reader.ReadInt16();

            // advance in the stream to skip the wave format block 
            len -= 16;                                                          // minimum format size
            while (len > 0)
            {
                Reader.ReadByte();
                len--;
            }

            // assume the data chunk is aligned
            while (m_stream.Position < m_stream.Length && ReadChunk(Reader) != "data")
                ;

            if (m_stream.Position >= m_stream.Length)
                throw new Exception("Invalid file format");

            m_length = Reader.ReadInt32();
            m_start_pos = m_stream.Position;
            m_position = 0;
        }

        /// <summary>
        /// Read wave samples
        /// </summary>
        /// <param name="count">Number of samples</param>
        /// <returns>Samples in byte format</returns>
        public byte[] ReadBytes(int count)
        {
            byte[] m_PlayBuffer = new byte[count];
            m_stream.Read(m_PlayBuffer, 0, count);
            m_stream.Position = m_start_pos;
            return m_PlayBuffer;
        }

        /// <summary>
        /// Read wave samples
        /// </summary>
        /// <param name="count">Number of samples</param>
        /// <returns>Samples in integer format</returns>
        public short[] ReadShort(int count)
        {
            Int16[] data_out;
            byte[] data = ReadBytes(count);

            if (data == null)
                return null;
            
            int index = 0;
            data_out = new Int16[data.Length / 2];
            for (int i = 0; i + 2 <= data.Length; i += 2)
            {
                Int16 d = BitConverter.ToInt16(data, i);
                data_out[index] = (Int16)(d == Int16.MinValue ? Int16.MinValue + 1 : d);
                index++;
            }

            return data_out;
        }

        /// <summary>
        /// Read wave samples.
        /// </summary>
        /// <param name="count">Number of samples.</param>
        /// <returns>Samples in floating point format.</returns>
        public float[] ReadFloat(int count)
        {
            float[] data_out;
            byte[] data = ReadBytes(count);

            if (data == null)
                return null;

            int index = 0;
            data_out = new float[data.Length / 2];
            for (int i = 0; i + 2 <= data.Length; i += 2)
            {
                float d = BitConverter.ToInt16(data, i);
                index++;
            }

            return data_out;
        }

        /// <summary>
        /// Write data into RIFF wave file
        /// </summary>
        /// <param name="filename">File name</param>
        /// <param name="data">Dat to write in byte format</param>
        /// <param name="format">Wave format of RIFF file</param>
        public static void Save(string filename, short[] data, WaveFormat format)
        {
            byte[] buff = ToByte(data);
            FileStream fs = null;
            try
            {
                fs = new FileStream(filename, FileMode.Create, FileAccess.Write);
                Recorder.WriteHeader(fs, format, buff.Length);
                fs.Write(buff, 0, buff.Length);
            }
            finally
            {
                fs.Close();
            }
        }

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


        #region Transcription
        /// <summary>
        /// Attach transcription of audio file.
        /// </summary>
        /// <param name="filename">File name.</param>
        public void AttachTranscriptionFile(string filename)
        {
            if (!File.Exists(filename))
                throw new Exception("File not found!");

            int lineCount = File.ReadAllLines(filename).Length;
            m_intervals = new Intervals[lineCount];

            using (TextReader reader = File.OpenText(filename))
            {
                int i = 0;
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    string[] item = line.Split('\t');
                    float start = float.Parse(item[1]) * m_format.nSamplesPerSec;
                    float stop = float.Parse(item[2]) * m_format.nSamplesPerSec;
                    m_intervals[i++] = new Intervals(item[0], (int)start, (int)stop);
                }
            }

            HasTranscription = true;
        }

        /// <summary>
        /// Search transcription file and return current feature
        /// </summary>
        /// <param name="ax">Index in audio file.</param>
        /// <returns>Current feature.</returns>
        public char GetTranscription(int x, int y)
        {
            for (int i = 0; i < m_intervals.Length; i++)
            {
                Point AB = new Point(m_intervals[i].Start, m_intervals[i].Stop);
                if (x >= AB.X && x <= AB.Y)
                {
                    int a = x - AB.X;
                    int b = y - AB.Y;
                    if (a >= b)
                        return m_intervals[i].Feature;
                    else
                        return m_intervals[i].Feature;
                }
            }
            return ' ';
        }
        #endregion

        #region Split channel
        public static void SplitChannels(byte[] wave, out byte[] _waveLeft, out byte[] _waveRight)
        {
            _waveLeft = new byte[wave.Length / 2];
            _waveRight = new byte[wave.Length / 2];

            // Split out channels from sample
            int h = 0;
            for (int i = 0; (i + 4) < wave.Length; i += 4)
            {
                Array.Copy(wave, i, _waveLeft, h, 2);
                Array.Copy(wave, i + 2, _waveRight, h, 2);
                h++;
            }
        }

        public static void SplitChannels(byte[] wave, out Int16[] _waveLeft, out Int16[] _waveRight)
        {
            _waveLeft = new Int16[wave.Length / 2 + 1];
            _waveRight = new Int16[wave.Length / 2 + 1];

            // Split out channels from sample
            int h = 0;
            for (int i = 0; (i + 2) < wave.Length; i += 2)
            {
                _waveLeft[h] = BitConverter.ToInt16(wave, i);
                _waveRight[h] = BitConverter.ToInt16(wave, i + 2);
                h++;
            }
        }

        public static void SplitChannels(byte[] wave, out float[] _waveLeft, out float[] _waveRight)
        {
            _waveLeft = new float[wave.Length / 2 + 1];
            _waveRight = new float[wave.Length / 2 + 1];

            // Split out channels from sample
            int h = 0;
            for (int i = 0; (i + 2) < wave.Length; i += 2)
            {
                _waveLeft[h] = BitConverter.ToSingle(wave, i);
                _waveRight[h] = BitConverter.ToSingle(wave, i + 2);
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

        /// <summary>
        /// Stop playback and close record stream.
        /// </summary>
        public void Dispose()
        {
            Stop();
        }

        ~WavPlayer()
        {
            Dispose();
            m_stream.Close();
        }
    }//end class
}//end namespace
