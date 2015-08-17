using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace GoodAI.Modules.SoundProcessing
{
    // Event delegates
    public delegate void ByteRecordingEventHandler(byte[] data);
    public delegate void ShortRecordingEventHandler(short[] data);
    public delegate void FloatRecordingEventHandler(float[] data);
    public delegate void RecordFinnishEventHandler();

    /// <summary>
    /// <para>Low level Audio recorder.</para>
    /// <para>Please add Dispose() method to FormClose() event.</para>
    /// </summary>
    public class Recorder : IDisposable
    {
        //const
        public int BUFF_SIZE = 4096;
        public const byte BUFF_CNT = 3;

        #region Declarations

        public Stream m_stream;
        public WaveFormat m_format;
        public string m_filename;
        public int m_device;

        public long m_length;
        public long m_position;

        // flags
        public bool is_recording = false;

        //var
        private Thread m_Thread;
        private IntPtr m_WavePtr;
        private WaveBuffer m_Buffers;                       // linked list
        private WaveBuffer m_CurrentBuffer;

        #endregion

        /// <summary>
        /// Internal class for translation of exception messages
        /// </summary>
        internal class WaveInHelper
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
            #region Declarations
            public WaveBuffer NextBuffer;

            private IntPtr m_Wave;
            private AutoResetEvent m_RecordEvent = new AutoResetEvent(false);

            private byte[] m_HeaderData;
            private WaveNative.WaveHdr m_Header;
            private GCHandle m_HeaderHandle;
            private GCHandle m_HeaderDataHandle;

            private bool m_Recording;
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
                    case WaveNative.MM_WIM_OPEN:
                        break;
                    case WaveNative.MM_WIM_DATA:
                        try
                        {
                            GCHandle h = (GCHandle)wavhdr.dwUser;
                            WaveBuffer buf = (WaveBuffer)h.Target;
                            buf.OnCompleted();
                        }
                        catch {
                            Thread.CurrentThread.Join();
                        }
                        break;
                    case WaveNative.MM_WIM_CLOSE:
                        break;
                }
            }

            public WaveBuffer(IntPtr waveInHandle, int size)
            {
                m_Wave = waveInHandle;

                m_HeaderHandle = GCHandle.Alloc(m_Header, GCHandleType.Pinned);
                m_Header.dwUser = (IntPtr)GCHandle.Alloc(this);
                m_HeaderData = new byte[size];
                m_HeaderDataHandle = GCHandle.Alloc(m_HeaderData, GCHandleType.Pinned);
                m_Header.lpData = m_HeaderDataHandle.AddrOfPinnedObject();
                m_Header.dwBufferLength = size;
                WaveInHelper.Try(WaveNative.waveInPrepareHeader(m_Wave, ref m_Header, Marshal.SizeOf(m_Header)));
            }

            public bool Read()
            {
                lock (this)
                {
                    m_RecordEvent.Reset();
                    m_Recording = WaveNative.waveInAddBuffer(m_Wave, ref m_Header, Marshal.SizeOf(m_Header)) == WaveNative.MMSYSERR_NOERROR;
                    return m_Recording;
                }
            }

            public void WaitFor()
            {
                if (m_Recording)
                    m_Recording = m_RecordEvent.WaitOne();
                else
                    Thread.Sleep(0);
            }

            private void OnCompleted()
            {
                m_RecordEvent.Set();
                m_Recording = false;
            }

            public void Dispose()
            {
                if (m_Header.lpData != IntPtr.Zero)
                {
                    WaveNative.waveInUnprepareHeader(m_Wave, ref m_Header, Marshal.SizeOf(m_Header));
                    m_HeaderHandle.Free();
                    m_Header.lpData = IntPtr.Zero;
                }
                m_RecordEvent.Close();
                if (m_HeaderDataHandle.IsAllocated)
                    m_HeaderDataHandle.Free();
                GC.SuppressFinalize(this);
            }

            ~WaveBuffer()
            {
                Dispose();
            }

        }//end class

        // callbacks
        private static WaveNative.WaveDelegate m_BufferProc = new WaveNative.WaveDelegate(WaveBuffer.ProcessMsg);

        #region events
        public event ByteRecordingEventHandler ByteRecording;
        public event ShortRecordingEventHandler ShortRecording;
        public event FloatRecordingEventHandler FloatRecording;
        public event RecordFinnishEventHandler RecordFinnish;

        protected virtual void OnByteRecord(byte[] data)
        {
            if (ByteRecording != null)
                ByteRecording(data);
        }

        protected virtual void OnShortRecord(short[] data)
        {
            if (ShortRecording != null)
                ShortRecording(data);
        }

        protected virtual void OnFloatRecord(float[] data)
        {
            if (FloatRecording != null)
                FloatRecording(data);
        }

        protected virtual void OnRecordFinnish()
        {
            if (RecordFinnish != null)
                RecordFinnish();
        }
        #endregion



        /// <summary>
        /// Low level Audio recorder.
        /// </summary>
        /// <param name="filename">Name of the file to store recorded data.</param>
        /// /// <param name="device_id">Id of device. Default value = -1.</param>
        public Recorder(WaveFormat format, int device_id, int buff_size)
        {
            if (format == null)
                throw new Exception("Invalid format.");
            
            m_format = format;
            BUFF_SIZE = buff_size;

            try
            {
                if (m_stream != null)
                    m_stream.Close();

                // Open recording device
                WaveInHelper.Try(WaveNative.waveInOpen(out m_WavePtr, device_id, m_format, m_BufferProc, 0, WaveNative.CALLBACK_FUNCTION));
                AllocateBuffers(BUFF_SIZE, BUFF_CNT);
                m_device = device_id;
            }
            catch { }
        }

        /// <summary>
        /// Low level Audio recorder.
        /// </summary>
        /// <param name="filename">Name of the file to store recorded data.</param>
        /// /// <param name="device_id">Id of device. Default value = -1.</param>
        public Recorder(string filename, WaveFormat format, int device_id, int buff_size) 
        {
            m_filename = filename;

            if (format == null)
                throw new Exception("Invalid format.");
            m_format = format;
            BUFF_SIZE = buff_size;

            try
            {
                if (m_stream != null)
                    m_stream.Close();

                // Open recording device
                WaveInHelper.Try(WaveNative.waveInOpen(out m_WavePtr, device_id, m_format, m_BufferProc, 0, WaveNative.CALLBACK_FUNCTION));
                AllocateBuffers(BUFF_SIZE, BUFF_CNT);
                m_device = device_id;
            }
            catch { }
        }

        /// <summary>
        /// Start recording
        /// </summary>
        public void Record()
        {
            // set recording flag
            if (is_recording)
                return;
            else
            {
                is_recording = true;

                // prepare stream
                if (m_stream != null)
                    m_stream.Close();

                // Open recording device
                WaveInHelper.Try(WaveNative.waveInOpen(out m_WavePtr, m_device, m_format, m_BufferProc, 0, WaveNative.CALLBACK_FUNCTION));
                AllocateBuffers(BUFF_SIZE, BUFF_CNT);
                
                
                try
                {
                    if (m_filename != null)
                    {
                        // Create or replace a file and write header
                        m_stream = new FileStream(m_filename, FileMode.Create, FileAccess.Write);
                        WriteHeader(m_stream, m_format, m_stream.Length);
                    }
                    else
                        m_stream = new MemoryStream();
                }
                catch (Exception e) { throw new Exception(e.Message); }
            }

            // prepare buffers
            for (int i = 0; i < BUFF_CNT; i++)
            {
                SelectNextBuffer();
                m_CurrentBuffer.Read();
            }

            //start recording
            WaveInHelper.Try(WaveNative.waveInStart(m_WavePtr));
            m_Thread = new Thread(new ThreadStart(ThreadRecord));
            m_Thread.Start();
        }

        /// <summary>
        /// Temporarily pause recording
        /// </summary>
        public void Pause()
        {
            is_recording = false;

            if (m_Thread != null)
            {
                m_Thread.Abort();
                m_Thread.Join();
            }
        }

        /// <summary>
        /// Stop recording
        /// </summary>
        public void Stop()
        {
            is_recording = false;

            if (m_Thread != null)
            {
                m_Thread.Abort();
                m_Thread.Join();
            }

            if (m_stream != null && m_stream.CanWrite)
            {
                m_stream.Position = 0;
                WriteHeader(m_stream, m_format, m_stream.Length);
                m_stream.Close();
            }

            OnRecordFinnish();
        }

        /// <summary>
        /// Get list of available recodring devices.
        /// </summary>
        /// <returns>List of names.</returns>
        public string[] GetDevicesList()
        {
            List<string> DevList = new List<string>();

            // get number of available input devices
            int waveInDevicesCount = WaveNative.waveInGetNumDevs();
            if (waveInDevicesCount > 0)
            {
                for (int uDeviceID = 0; uDeviceID < waveInDevicesCount; uDeviceID++)
                {
                    WaveNative.WaveInCaps waveInCaps = new WaveNative.WaveInCaps();
                    WaveNative.waveInGetDevCaps(uDeviceID, ref waveInCaps, Marshal.SizeOf(typeof(WaveNative.WaveInCaps)));
                    DevList.Add(new string(waveInCaps.szPname).Remove(
                                new string(waveInCaps.szPname).IndexOf('\0')).Trim());
                }
            }
            return DevList.ToArray();
        }

        private static byte[] WriteChunk(string chunk)
        {
            byte[] ch = new byte[4];
            ch = Encoding.ASCII.GetBytes(chunk);
            return ch;
        }

        /// <summary>
        /// Write RIFF header to the stream.
        /// </summary>
        /// <param name="stream">Source stream.</param>
        /// <param name="format">Wave format.</param>
        public static void WriteHeader(Stream stream, WaveFormat format, long stream_length)
        {
            if (stream == null)
                throw new Exception("No stream available.");

            if (!stream.CanWrite)
                throw new Exception("Cannot write into stream.");

            if (format == null)
                throw new Exception("No format specified.");

            BinaryWriter Writer = new BinaryWriter(stream);
            Writer.Write(WriteChunk("RIFF"));                       //chunk descriptor
            Writer.Write((Int32)2048);                              //chunk size = 2048
            Writer.Write(WriteChunk("WAVE"));
            Writer.Write(WriteChunk("fmt "));                       //fmt subchunk
            Writer.Write((Int32)16);                                //subchunk1 size = 16

            Writer.Write(format.wFormatTag);                        //audio format = 1(PCM)
            Writer.Write(format.nChannels);                         //count of channels
            Writer.Write(format.nSamplesPerSec);                    //sample rate
            Writer.Write(format.nAvgBytesPerSec);                   //byte rate
            Writer.Write(format.nBlockAlign);                       //block align
            Writer.Write(format.wBitsPerSample);                    //bits per sample

            Writer.Write(WriteChunk("data"));                       //data chunk
            Writer.Write(stream_length);                            //subchunk2 size
            //data
        }

        private void ThreadRecord()
        {
            while (is_recording)
            {
                Advance();
                byte[] m_RecBuffer = new byte[BUFF_SIZE];
                Marshal.Copy(m_CurrentBuffer.Data, m_RecBuffer, 0, BUFF_SIZE);
                if (m_stream != null)
                    m_stream.Write(m_RecBuffer, 0, BUFF_SIZE);
                
                m_CurrentBuffer.Read();

                // access low level recorded data
                OnByteRecord(m_RecBuffer);                                      //byte

                short[] data = new short[m_RecBuffer.Length / 2];               //short
                Buffer.BlockCopy(m_RecBuffer, 0, data, 0, m_RecBuffer.Length);
                OnShortRecord(data);

                float[] fdata = new float[data.Length];
                for (int i = 0; i < data.Length; i++)
                    fdata[i] = (float)data[i];

                OnFloatRecord(fdata);
            }
        }

        private void Advance()
        {
            SelectNextBuffer();
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

        private void SelectNextBuffer()
        {
            m_CurrentBuffer = m_CurrentBuffer == null ? m_Buffers : m_CurrentBuffer.NextBuffer;
        }

        /// <summary>
        /// Stop recording and close record stream.
        /// </summary>
        public void Dispose()
        {
            Stop();
        }

        ~Recorder()
        {
            Dispose();
            if (m_stream != null)
                m_stream.Close();
        }

    }//end class
}//end namespace
