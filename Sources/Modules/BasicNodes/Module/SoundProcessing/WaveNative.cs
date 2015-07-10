using System;
using System.Runtime.InteropServices;

namespace AudioLib
{
    /// <summary>
    /// Audio framework library wrapper.
    /// </summary>
	internal class WaveNative
	{
		// consts
        private const string mmdll = "winmm.dll";

		public const int MMSYSERR_NOERROR = 0;              // no error

		public const int MM_WOM_OPEN = 0x3BB;
		public const int MM_WOM_CLOSE = 0x3BC;
		public const int MM_WOM_DONE = 0x3BD;

		public const int MM_WIM_OPEN = 0x3BE;
		public const int MM_WIM_CLOSE = 0x3BF;
		public const int MM_WIM_DATA = 0x3C0;

		public const int CALLBACK_FUNCTION = 0x00030000;    // dwCallback is a FARPROC 

		public const int TIME_MS = 0x0001;                  // time in milliseconds 
		public const int TIME_SAMPLES = 0x0002;             // number of wave samples 
		public const int TIME_BYTES = 0x0004;               // current byte offset 

        public const int SND_FILENAME = 0x00020000;         // name is file name
        public const int SND_ASYNC = 0x0001;                // play asynchronously

		// callbacks
		public delegate void WaveDelegate(IntPtr hdrvr, int uMsg, int dwUser, ref WaveHdr wavhdr, int dwParam2);

		// structs 
		[StructLayout(LayoutKind.Sequential)] public struct WaveHdr
		{
            /// <summary>pointer to locked data buffer</summary>
			public IntPtr lpData;
            /// <summary>length of data buffer</summary>
            public int dwBufferLength;
            /// <summary>used for input only</summary>
			public int dwBytesRecorded;
            /// <summary>for client's use</summary>
			public IntPtr dwUser;
            /// <summary>assorted flags (see defines)</summary>
			public int dwFlags;
            /// <summary>loop control counter</summary>
			public int dwLoops;
            /// <summary>PWaveHdr, reserved for driver</summary>
			public IntPtr lpNext;
            /// <summary>reserved for driver</summary>
			public int reserved; 
		}

        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct WaveInCaps
        {
            public short wMid;
            public short wPid;
            public int vDriverVersion;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 32)]
            public char[] szPname;
            public uint dwFormats;
            public short wChannels;
            public short wReserved1;
        }

        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct WaveOutCaps
        {
            public short wMid;
            public short wPid;
            public int vDriverVersion;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 32)]
            public char[] szPname;
            public uint dwFormats;
            public short wChannels;
            public short wReserved1;
            public uint dwSupport;
        } 


        // WaveOut calls
		/// <summary>
        /// The waveOutGetNumDevs function retrieves the number of waveform-audio output devices present in the system.
		/// </summary>
        /// <returns>Returns the number of devices. A return value of zero means that no devices are present or that an error occurred.</returns>
		[DllImport(mmdll)]
		public static extern int waveOutGetNumDevs();
        /// <summary>
        /// The waveOutGetDevCaps function retrieves the capabilities of a given waveform-audio output device.
        /// </summary>
        /// <param name="uDeviceID">
        ///                   <para>Identifier of the waveform-audio output device. </para>
        ///                   <para>It can be either a device identifier or a handle of an open waveform-audio output device.</para>
        /// </param>
        /// <param name="lpCaps">Pointer to a WAVEOUTCAPS structure to be filled with information about the capabilities of the device. </param>
        /// <param name="uSize">Size, in bytes, of the WAVEOUTCAPS structure.</param>
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///    <para>MMSYSERR_BADDEVICEID 	Specified device identifier is out of range.</para>
        ///    <para>MMSYSERR_NODRIVER 	No device driver is present.</para>
        ///    <para>MMSYSERR_NOMEM 	Unable to allocate or lock memory.</para>
        /// </returns>
        [DllImport(mmdll)]
        public static extern int waveOutGetDevCaps(int uDeviceID, ref WaveOutCaps lpCaps, int uSize);
        /// <summary>
        /// The waveOutPrepareHeader function prepares a waveform-audio data block for playback.
        /// </summary>
        /// <param name="hWaveOut">Handle to the waveform-audio output device.</param>
        /// <param name="lpWaveOutHdr">Pointer to a WaveHdr structure that identifies the data block to be prepared.</param>
        /// <param name="uSize">Size, in bytes, of the WaveHdr structure.</param>
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.</returns>
		[DllImport(mmdll)]
		public static extern int waveOutPrepareHeader(IntPtr hWaveOut, ref WaveHdr lpWaveOutHdr, int uSize);
        /// <summary>
        /// <para>The waveOutUnprepareHeader function cleans up the preparation performed by the waveOutPrepareHeader function.</para>
        /// <para>This function must be called after the device driver is finished with a data block. You must call this function before freeing the buffer.</para>
        /// </summary>
        /// <param name="hWaveOut">Handle to the waveform-audio output device.</param>
        /// <param name="lpWaveOutHdr">Pointer to a WaveHdr structure identifying the data block to be cleaned up.</param>
        /// <param name="uSize">Size, in bytes, of the WaveHdr structure.</param>
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.</returns>
		[DllImport(mmdll)]
		public static extern int waveOutUnprepareHeader(IntPtr hWaveOut, ref WaveHdr lpWaveOutHdr, int uSize);
        /// <summary>
        /// The waveOutWrite function sends a data block to the given waveform-audio output device.
        /// </summary>
        /// <param name="hWaveOut">Handle to the waveform-audio output device.</param>
        /// <param name="lpWaveOutHdr">Pointer to a WaveHdr structure containing information about the data block.</param>
        /// <param name="uSize">Size, in bytes, of the WaveHdr structure.</param>
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.</returns>
		[DllImport(mmdll)]
		public static extern int waveOutWrite(IntPtr hWaveOut, ref WaveHdr lpWaveOutHdr, int uSize);
        /// <summary>
        /// The waveOutOpen function opens the given waveform-audio output device for playback.
        ///</summary>
        ///
        /// <param name="hWaveOut">Pointer to a buffer that receives a handle identifying the open waveform-audio output device. 
        ///                  <para>Use the handle to identify the device when calling other waveform-audio output functions.</para>
        ///                  <para>This parameter might be NULL if the WAVE_FORMAT_QUERY flag is specified for fdwOpen.</para></param>
        ///                  
        /// <param name="uDeviceID">Identifier of the waveform-audio output device to open. 
        ///                   <para>It can be either a device identifier or a handle of an open waveform-audio input device.</para>
        ///                   <para>You can use the following flag instead of a device identifier.</para>
        ///                   <para>WAVE_MAPPER : The function selects a waveform-audio output device capable of playing the given format.</para></param>
        ///                   
        /// <param name="lpFormat">Pointer to a WAVEFORMATEX structure that identifies the format of the waveform-audio data to be sent to the device. 
        ///                  <para>You can free this structure immediately after passing it to waveOutOpen.</para></param>
        ///                  
        /// <param name="dwCallback">Pointer to a fixed callback function, an event handle, a handle to a window, 
        ///                    <para>or the identifier of a thread to be called during waveform-audio playback to</para>
        ///                    <para>process messages related to the progress of the playback.</para>
        ///                    <para>If no callback function is required, this value can be zero.</para>
        ///                    <para>For more information on the callback function, see waveOutProc.</para></param>
        ///                    
        /// <param name="dwInstance">User-instance data passed to the callback mechanism. 
        ///                    <para>This parameter is not used with the window callback mechanism.</para></param>
        ///                    
        /// <param name="dwFlags">Flags for opening the device. The following values are defined.
        ///                 <para>CALLBACK_EVENT 	    :The dwCallback parameter is an event handle.</para>
        ///                 <para>CALLBACK_FUNCTION 	:The dwCallback parameter is a callback procedure address.</para>
        ///                 <para>CALLBACK_NULL 	    :No callback mechanism. This is the default setting.</para>
        ///                 <para>CALLBACK_THREAD 	    :The dwCallback parameter is a thread identifier.</para>
        ///                 <para>CALLBACK_WINDOW 	    :The dwCallback parameter is a window handle.</para>
        ///                 <para>WAVE_ALLOWSYNC 	    :If this flag is specified, a synchronous waveform-audio device can be opened.</para>
        ///                 <para>                       If this flag is not specified while opening a synchronous driver, the device will fail to open.</para>
        ///                 <para>WAVE_FORMAT_DIRECT 	:If this flag is specified, the ACM driver does not perform conversions on the audio data.</para>
        ///                 <para>WAVE_FORMAT_QUERY 	:If this flag is specified, waveOutOpen queries the device to determine if it supports the given format,</para>
        ///                 <para>                       but the device is not actually opened.</para>
        ///                 <para>WAVE_MAPPED 	        :If this flag is specified, the uDeviceID parameter specifies a waveform-audio device to be mapped to by the wave mapper.</para></param>
        ///                 
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///                 <para>MMSYSERR_ALLOCATED 	:Specified resource is already allocated.</para>
        ///                 <para>MMSYSERR_BADDEVICEID :Specified device identifier is out of range.</para>
        ///                 <para>MMSYSERR_NODRIVER 	:No device driver is present.</para>
        ///                 <para>MMSYSERR_NOMEM 	    :Unable to allocate or lock memory.</para>
        ///                 <para>WAVERR_BADFORMAT 	:Attempted to open with an unsupported waveform-audio format.</para>
        ///                 <para>WAVERR_SYNC 	        :The device is synchronous but waveOutOpen was called without using the WAVE_ALLOWSYNC flag.</para>
        /// </returns>
		[DllImport(mmdll)]
		public static extern int waveOutOpen(out IntPtr hWaveOut, int uDeviceID, WaveFormat lpFormat, WaveDelegate dwCallback, int dwInstance, int dwFlags);
        /// <summary>
        /// The waveOutReset function stops playback on the given waveform-audio output device and resets the current position to zero. 
        /// <para>All pending playback buffers are marked as done and returned to the application.</para>
        /// </summary>
        /// 
        /// <param name="hWaveOut">Handle to the waveform-audio output device.</param>
        /// 
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///    <para>MMSYSERR_INVALHANDLE 	:Specified device handle is invalid.</para>
        ///    <para>MMSYSERR_NODRIVER 	    :No device driver is present.</para>
        ///    <para>MMSYSERR_NOMEM 	    :Unable to allocate or lock memory.</para>
        ///    <para>MMSYSERR_NOTSUPPORTED  :Specified device is synchronous and does not support pausing.</para>
        /// </returns>
		[DllImport(mmdll)]
		public static extern int waveOutReset(IntPtr hWaveOut);
        /// <summary>
        /// The waveOutClose function closes the given waveform-audio output device.
        /// </summary>
        /// <param name="hWaveOut">Handle to the waveform-audio output device. 
        /// <para>If the function succeeds, the handle is no longer valid after this call.</para></param>
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///    <para>MMSYSERR_INVALHANDLE 	:Specified device handle is invalid.</para>
        ///    <para>MMSYSERR_NODRIVER 	    :No device driver is present.</para>
        ///    <para>MMSYSERR_NOMEM 	    :Unable to allocate or lock memory.</para>
        ///    <para>MMSYSERR_NOTSUPPORTED  :Specified device is synchronous and does not support pausing.</para>
        /// </returns>
		[DllImport(mmdll)]
		public static extern int waveOutClose(IntPtr hWaveOut);
        /// <summary>
        /// <para>The waveOutPause function pauses playback on the given waveform-audio output device.</para>
        /// <para>The current position is saved. Use the waveOutRestart function to resume playback from the current position.</para>
        /// </summary>
        /// 
        /// <param name="hWaveOut">Handle to the waveform-audio output device.</param>
        /// 
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///    <para>MMSYSERR_INVALHANDLE 	:Specified device handle is invalid.</para>
        ///    <para>MMSYSERR_NODRIVER 	    :No device driver is present.</para>
        ///    <para>MMSYSERR_NOMEM 	    :Unable to allocate or lock memory.</para>
        ///    <para>MMSYSERR_NOTSUPPORTED  :Specified device is synchronous and does not support pausing.</para>
        /// </returns>
		[DllImport(mmdll)]
		public static extern int waveOutPause(IntPtr hWaveOut);
        /// <summary>
        /// The waveOutRestart function resumes playback on a paused waveform-audio output device.
        /// </summary>
        /// 
        /// <param name="hWaveOut">Handle to the waveform-audio output device.</param>
        /// 
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///    <para>MMSYSERR_INVALHANDLE 	:Specified device handle is invalid.</para>
        ///    <para>MMSYSERR_NODRIVER 	    :No device driver is present.</para>
        ///    <para>MMSYSERR_NOMEM 	    :Unable to allocate or lock memory.</para>
        ///    <para>MMSYSERR_NOTSUPPORTED  :Specified device is synchronous and does not support pausing.</para></returns>
		[DllImport(mmdll)]
		public static extern int waveOutRestart(IntPtr hWaveOut);
        /// <summary>
        /// The waveOutGetPosition function retrieves the current playback position of the given waveform-audio output device.
        /// </summary>
        /// <param name="hWaveOut">Handle to the waveform-audio output device.</param>
        /// <param name="lpInfo">Pointer to an MMTIME structure.</param>
        /// <param name="uSize">Size, in bytes, of the MMTIME structure.</param>
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///    <para>MMSYSERR_INVALHANDLE 	:Specified device handle is invalid.</para>
        ///    <para>MMSYSERR_NODRIVER 	    :No device driver is present.</para>
        ///    <para>MMSYSERR_NOMEM 	    :Unable to allocate or lock memory.</para>
        /// </returns>
		[DllImport(mmdll)]
		public static extern int waveOutGetPosition(IntPtr hWaveOut, out int lpInfo, int uSize);
        /// <summary>
        /// The waveOutSetVolume function sets the volume level of the specified waveform-audio  output device.
        /// </summary>
        /// <param name="hWaveOut">Handle to an open waveform-audio output device. This parameter can also be a device identifier.</param>
        /// <param name="dwVolume">
        ///                  <para>New volume setting. The low-order word contains the left-channel volume setting, and the high-order word contains the right-channel setting.</para>
        ///                  <para>A value of 0xFFFF represents full volume, and a value of 0x0000 is silence.</para>
        /// </param>
        /// 
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///    <para>MMSYSERR_INVALHANDLE 	:Specified device handle is invalid.</para>
        ///    <para>MMSYSERR_NODRIVER 	    :No device driver is present.</para>
        ///    <para>MMSYSERR_NOMEM 	    :Unable to allocate or lock memory.</para>
        ///    <para>MMSYSERR_NOTSUPPORTED  :Specified device is synchronous and does not support pausing.</para>
        /// </returns>
		[DllImport(mmdll)]
		public static extern int waveOutSetVolume(IntPtr hWaveOut, int dwVolume);
        /// <summary>
        /// The waveOutGetVolume function retrieves the current volume level of the specified waveform-audio output device.
        /// </summary>
        /// <param name="hWaveOut">Handle to an open waveform-audio output device. This parameter can also be a device identifier.</param>
        /// <param name="dwVolume">
        ///                 <para>Pointer to a variable to be filled with the current volume setting.</para>
        ///                 <para>The low-order word of this location contains the left-channel volume setting, </para>
        ///                 <para>and the high-order word contains the right-channel setting.</para>
        ///                 <para>A value of 0xFFFF represents full volume, and a value of 0x0000 is silence.</para>
        ///                 <para>If a device does not support both left and right volume control, </para>
        ///                 <para>the low-order word of the specified location contains the mono volume level.</para>
        ///                 <para>The full 16-bit setting(s) set with the waveOutSetVolume function is returned,</para>
        ///                 <para>regardless of whether the device supports the full 16 bits of volume-level control.</para>
        /// </param>
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///    <para>MMSYSERR_INVALHANDLE 	:Specified device handle is invalid.</para>
        ///    <para>MMSYSERR_NODRIVER 	    :No device driver is present.</para>
        ///    <para>MMSYSERR_NOMEM 	    :Unable to allocate or lock memory.</para>
        ///    <para>MMSYSERR_NOTSUPPORTED  :Specified device is synchronous and does not support pausing.</para>
        /// </returns>
		[DllImport(mmdll)]
		public static extern int waveOutGetVolume(IntPtr hWaveOut, out int dwVolume);

		// WaveIn calls
        /// <summary>
        /// The waveInGetNumDevs function returns the number of waveform-audio input devices present in the system.
        /// </summary>
        /// <returns>Returns the number of devices. A return value of zero means that no devices are present or that an error occurred.</returns>
		[DllImport(mmdll)]
		public static extern int waveInGetNumDevs();
        /// <summary>
        /// The waveInGetDevCaps function retrieves the capabilities of a given waveform-audio input device.
        /// </summary>
        /// <param name="uDeviceID">
        ///                   <para>Identifier of the waveform-audio output device. </para>
        ///                   <para>It can be either a device identifier or a handle of an open waveform-audio input device.</para>
        /// </param>
        /// <param name="lpCaps">Pointer to a WAVEINCAPS structure to be filled with information about the capabilities of the device.</param>
        /// <param name="uSize">Size, in bytes, of the WAVEINCAPS structure.</param>
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///    <para>MMSYSERR_BADDEVICEID 	:Specified device identifier is out of range.</para>
        ///    <para>MMSYSERR_NODRIVER 	    :No device driver is present.</para>
        ///    <para>MMSYSERR_NOMEM 	    :Unable to allocate or lock memory.</para>
        /// </returns>
        [DllImport(mmdll)]
        public static extern int waveInGetDevCaps(int uDeviceID, ref WaveInCaps lpCaps, int uSize);
        /// <summary>
        /// The waveInAddBuffer function sends an input buffer to the given waveform-audio input device. When the buffer is filled, the application is notified.
        /// </summary>
        /// <param name="hwi">Handle to the waveform-audio input device.</param>
        /// <param name="pwh">Pointer to a WAVEHDR structure that identifies the buffer.</param>
        /// <param name="cbwh">Size, in bytes, of the WAVEHDR structure.</param>
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///    <para>MMSYSERR_INVALHANDLE 	:Specified device handle is invalid.</para>
        ///    <para>MMSYSERR_NODRIVER 	    :No device driver is present.</para>
        ///    <para>MMSYSERR_NOMEM 	    :Unable to allocate or lock memory.</para>
        ///    <para>MMSYSERR_NOTSUPPORTED  :Specified device is synchronous and does not support pausing.</para>
        /// </returns>
		[DllImport(mmdll)]
		public static extern int waveInAddBuffer(IntPtr hwi, ref WaveHdr pwh, int cbwh);
        /// <summary>
        /// The waveInClose function closes the given waveform-audio input device.
        /// </summary>
        /// <param name="hwi">Handle to the waveform-audio input device. If the function succeeds, the handle is no longer valid after this call.</param>
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///    <para>MMSYSERR_INVALHANDLE 	:Specified device handle is invalid.</para>
        ///    <para>MMSYSERR_NODRIVER 	    :No device driver is present.</para>
        ///    <para>MMSYSERR_NOMEM 	    :Unable to allocate or lock memory.</para>
        ///    <para>WAVERR_STILLPLAYING 	:There are still buffers in the queue.</para>
        /// </returns>
		[DllImport(mmdll)]
		public static extern int waveInClose(IntPtr hwi);
        /// <summary>
        /// The waveInOpen function opens the given waveform-audio input device for recording.
        /// </summary>
        /// <param name="phwi">
        ///              <para>Pointer to a buffer that receives a handle identifying the open waveform-audio input device.</para>
        ///              <para>Use this handle to identify the device when calling other waveform-audio input functions. </para>
        ///              <para>This parameter can be NULL if WAVE_FORMAT_QUERY is specified for fdwOpen.</para></param>
        /// <param name="uDeviceID">
        ///                   <para>Identifier of the waveform-audio input device to open. It can be either a device identifier or a handle of an open waveform-audio input device. </para>
        ///                   <para>You can use the following flag instead of a device identifier.</para>
        ///                   <para>WAVE_MAPPER 	:The function selects a waveform-audio input device capable of recording in the specified format.</para>
        /// </param>
        /// <param name="lpFormat">Pointer to a WAVEFORMATEX structure that identifies the desired format for recording waveform-audio data. 
        ///                        You can free this structure immediately after waveInOpen returns.</param>
        /// <param name="dwCallback">
        ///                    <para>Pointer to a fixed callback function, an event handle, a handle to a window, </para>
        ///                    <para>or the identifier of a thread to be called during waveform-audio recording to process messages related to the progress of recording. </para>
        ///                    <para>If no callback function is required, this value can be zero. For more information on the callback function, see waveInProc.</para>
        /// </param>
        /// <param name="dwInstance">User-instance data passed to the callback mechanism. This parameter is not used with the window callback mechanism.</param>
        /// <param name="dwFlags">
        ///                 <para>Flags for opening the device. The following values are defined.</para>
        ///                 <para>CALLBACK_EVENT 	    :The dwCallback parameter is an event handle.</para>
        ///                 <para>CALLBACK_FUNCTION 	:The dwCallback parameter is a callback procedure address.</para>
        ///                 <para>CALLBACK_NULL 	    :No callback mechanism. This is the default setting.</para>
        ///                 <para>CALLBACK_THREAD 	    :The dwCallback parameter is a thread identifier.</para>
        ///                 <para>CALLBACK_WINDOW 	    :The dwCallback parameter is a window handle.</para>
        ///                 <para>WAVE_FORMAT_DIRECT 	:If this flag is specified, the ACM driver does not perform conversions on the audio data.</para>
        ///                 <para>WAVE_FORMAT_QUERY 	:The function queries the device to determine whether it supports the given format, but it does not open the device.</para>
        ///                 <para>WAVE_MAPPED 	        :The uDeviceID parameter specifies a waveform-audio device to be mapped to by the wave mapper.</para>
        /// </param>
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///    <para>MMSYSERR_ALLOCATED 	:Specified resource is already allocated.</para>
        ///    <para>MMSYSERR_BADDEVICEID 	:Specified device identifier is out of range.</para>
        ///    <para>MMSYSERR_NODRIVER 	    :No device driver is present.</para>
        ///    <para>MMSYSERR_NOMEM 	    :Unable to allocate or lock memory.</para>
        ///    <para>WAVERR_BADFORMAT 	    :Attempted to open with an unsupported waveform-audio format.</para>
        /// </returns>
		[DllImport(mmdll)]
		public static extern int waveInOpen(out IntPtr phwi, int uDeviceID, WaveFormat lpFormat, WaveDelegate dwCallback, int dwInstance, int dwFlags);
        /// <summary>
        /// The waveInPrepareHeader function prepares a buffer for waveform-audio input.
        /// </summary>
        /// <param name="hWaveIn">Handle to the waveform-audio input device.</param>
        /// <param name="lpWaveInHdr">Pointer to a WAVEHDR structure that identifies the buffer to be prepared.</param>
        /// <param name="uSize">Size, in bytes, of the WAVEHDR structure.</param>
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///    <para>MMSYSERR_INVALHANDLE 	:Specified device handle is invalid.</para>
        ///    <para>MMSYSERR_NODRIVER 	    :No device driver is present.</para>
        ///    <para>MMSYSERR_NOMEM 	    :Unable to allocate or lock memory.</para>
        /// </returns>
		[DllImport(mmdll)]
		public static extern int waveInPrepareHeader(IntPtr hWaveIn, ref WaveHdr lpWaveInHdr, int uSize);
        /// <summary>
        ///    <para>The waveInUnprepareHeader function cleans up the preparation performed by the waveInPrepareHeader function.</para>
        ///    <para>This function must be called after the device driver fills a buffer and returns it to the application.</para>
        ///    <para>You must call this function before freeing the buffer.</para>
        /// </summary>
        /// <param name="hWaveIn">Handle to the waveform-audio input device.</param>
        /// <param name="lpWaveInHdr">Pointer to a WAVEHDR structure identifying the buffer to be cleaned up.</param>
        /// <param name="uSize">Size, in bytes, of the WAVEHDR structure.</param>
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///    <para>MMSYSERR_INVALHANDLE 	:Specified device handle is invalid.</para>
        ///    <para>MMSYSERR_NODRIVER 	    :No device driver is present.</para>
        ///    <para>MMSYSERR_NOMEM 	    :Unable to allocate or lock memory.</para>
        ///    <para>WAVERR_STILLPLAYING 	:The buffer pointed to by the pwh parameter is still in the queue.</para>
        /// </returns>
		[DllImport(mmdll)]
		public static extern int waveInUnprepareHeader(IntPtr hWaveIn, ref WaveHdr lpWaveInHdr, int uSize);
        /// <summary>
        /// <para>The waveInReset function stops input on the given waveform-audio input device and resets the current position to zero.</para>
        /// <para>All pending buffers are marked as done and returned to the application.</para>
        /// </summary>
        /// <param name="hwi">Handle to the waveform-audio input device.</param>
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///    <para>MMSYSERR_INVALHANDLE 	:Specified device handle is invalid.</para>
        ///    <para>MMSYSERR_NODRIVER 	    :No device driver is present.</para>
        ///    <para>MMSYSERR_NOMEM 	    :Unable to allocate or lock memory.</para>
        /// </returns>
		[DllImport(mmdll)]
		public static extern int waveInReset(IntPtr hwi);
        /// <summary>
        /// The waveInStart function starts input on the given waveform-audio input device.
        /// </summary>
        /// <param name="hwi">Handle to the waveform-audio input device.</param>
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///    <para>MMSYSERR_INVALHANDLE 	:Specified device handle is invalid.</para>
        ///    <para>MMSYSERR_NODRIVER 	    :No device driver is present.</para>
        ///    <para>MMSYSERR_NOMEM 	    :Unable to allocate or lock memory.</para>
        /// </returns>
		[DllImport(mmdll)]
		public static extern int waveInStart(IntPtr hwi);
        /// <summary>
        /// The waveInStop function stops waveform-audio input.
        /// </summary>
        /// <param name="hwi">Handle to the waveform-audio input device.</param>
        /// <returns>Returns MMSYSERR_NOERROR if successful or an error otherwise. Possible error values include the following.
        ///    <para>MMSYSERR_INVALHANDLE 	:Specified device handle is invalid.</para>
        ///    <para>MMSYSERR_NODRIVER 	    :No device driver is present.</para>
        ///    <para>MMSYSERR_NOMEM 	    :Unable to allocate or lock memory.</para>
        /// </returns>
		[DllImport(mmdll)]
		public static extern int waveInStop(IntPtr hwi);


        // other calls
        [DllImport(mmdll)]
        public static extern bool PlaySound(string pszSound, UIntPtr hmod, uint fdwSound);
	}//end class
}//end namespace
