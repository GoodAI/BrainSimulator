using System;
using System.Drawing;
using System.IO;
using System.Text;

namespace GoodAI.Modules.SoundProcessing
{
    // Event delegates
    public delegate void BytePlaybackEventHandler(byte[] data);
    public delegate void ShortPlaybackEventHandler(short[] data);
    public delegate void PositionChangeEventHandler(long position);
    public delegate void PlaybackFinnishEventHandler();

    public class WavPlayer : IDisposable
    {
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

        #endregion

        #region Declarations

        public Stream m_stream;
        
        public int m_length;
        public long m_position;
        private long m_start_pos;

        // flags
        public bool is_playing = false;

        // Format
        public short m_FormatTag;
        public short m_NumChannels = 1;
        public int m_SamplesPerSec = 44100;
        public int m_AvgBytesPerSec;
        public short m_NumBlockAlign;
        public short m_BitsPerSample = 16;

        private Intervals[] m_intervals;

        #endregion


        /// <summary>
        /// Low-level .wav format audio player.
        /// </summary>
        /// <param name="filename">Path to audio file to play.</param>
        public WavPlayer(string filename) 
        {
            if (!File.Exists(filename))
                throw new Exception("File not found!");

            try
            {
                // Open file, read header and open playback device
                m_stream = new FileStream(filename, FileMode.Open, FileAccess.Read);
                ReadHeader();
                m_start_pos = m_stream.Position;
            }
            catch { }
        }

        /// <summary>
        /// Low level Audio player.
        /// </summary>
        public WavPlayer(Stream stream)
        {
            if (stream == null)
                throw new Exception("Invalid stream!");

            try
            {
                // Set the buf size, open stream, prepare format and open playback device
                m_stream = stream;
                m_start_pos = 0;
                m_length = (int)stream.Length;
            }
            catch (Exception e) { throw new Exception(e.Message); }
        }

        private string ReadChunk(BinaryReader reader)
        {
            byte[] ch = new byte[4];
            reader.Read(ch, 0, ch.Length);
            return Encoding.ASCII.GetString(ch);
        }

        private byte[] WriteChunk(string chunk)
        {
            byte[] ch = new byte[4];
            ch = Encoding.ASCII.GetBytes(chunk);
            return ch;
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

            m_FormatTag = Reader.ReadInt16();
            m_NumChannels = Reader.ReadInt16();
            m_SamplesPerSec = Reader.ReadInt32();
            m_AvgBytesPerSec = Reader.ReadInt32();
            m_NumBlockAlign = Reader.ReadInt16();
            m_BitsPerSample = Reader.ReadInt16();

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
        /// Write RIFF header to the stream.
        /// </summary>
        /// <param name="stream">Source stream.</param>
        /// <param name="stream_length">Stream length.</param>
        public void WriteHeader(Stream stream, long stream_length)
        {
            if (stream == null)
                throw new Exception("No stream available.");

            if (!stream.CanWrite)
                throw new Exception("Cannot write into stream.");

            BinaryWriter Writer = new BinaryWriter(stream);
            Writer.Write(WriteChunk("RIFF"));                       //chunk descriptor
            Writer.Write((Int32)2048);                              //chunk size = 2048
            Writer.Write(WriteChunk("WAVE"));
            Writer.Write(WriteChunk("fmt "));                       //fmt subchunk
            Writer.Write((Int32)16);                                //subchunk1 size = 16

            Writer.Write(m_FormatTag);                        //audio format = 1(PCM)
            Writer.Write(m_NumChannels);                         //count of channels
            Writer.Write(m_SamplesPerSec);                    //sample rate
            Writer.Write(m_AvgBytesPerSec);                   //byte rate
            Writer.Write(m_NumBlockAlign);                       //block align
            Writer.Write(m_BitsPerSample);                    //bits per sample

            Writer.Write(WriteChunk("data"));                       //data chunk
            Writer.Write(stream_length);                            //subchunk2 size
            //data
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
        public void Save(string filename, short[] data)
        {
            byte[] buff = ToByte(data);
            FileStream fs = null;
            try
            {
                fs = new FileStream(filename, FileMode.Create, FileAccess.Write);
                WriteHeader(fs, buff.Length);
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
                    float start = float.Parse(item[1]) * m_SamplesPerSec;
                    float stop = float.Parse(item[2]) * m_SamplesPerSec;
                    m_intervals[i++] = new Intervals(item[0], (int)start, (int)stop);
                }
            }

            HasTranscription = true;
        }

        /// <summary>
        /// Attach transcription of audio file.
        /// </summary>
        /// <param name="filename">File name.</param>
        public void AttachTranscription(string transcription)
        {
            if (transcription == "")
                return;

            string[] line = transcription.Split('\n');
            m_intervals = new Intervals[line.Length];
            for (int i = 0; i < line.Length; i++)
            {
                string[] item = line[i].Split('\t');
                float start = float.Parse(item[1]) * m_SamplesPerSec;
                float stop = float.Parse(item[2]) * m_SamplesPerSec;
                m_intervals[i] = new Intervals(item[0], (int)start, (int)stop);
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

        public void Dispose()
        {
            try
            {
                m_stream.Close();
            }catch(Exception e)
            {
                m_stream = null;
            }
        }

        ~WavPlayer()
        {
            Dispose();
        }

        
    }//end class
}//end namespace
