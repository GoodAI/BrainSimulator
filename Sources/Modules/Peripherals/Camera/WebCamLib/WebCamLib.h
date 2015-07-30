//*****************************************************************************************
//  File:       WebCamLib.h
//  Project:    WebcamLib
//  Author(s):  John Conwell
//              Gary Caldwell
//
//  Declares the webcam DirectShow wrapper used by TouchlessLib
//*****************************************************************************************

#pragma once

using namespace System;
using namespace System::Runtime::InteropServices;

namespace WebCamLib
{
	/// <summary>
	/// Store webcam name, index
	/// </summary>
	public ref class CameraInfo
	{
	public:
		property int index;
		property String^ name;
	};

	/// <summary>
	/// DirectShow wrapper around a web cam, used for image capture
	/// </summary>
	public ref class CameraMethods
	{
	public:
		/// <summary>
		/// Initializes information about all web cams connected to machine
		/// </summary>
		CameraMethods();

		/// <summary>
		/// Delegate used by DirectShow to pass back captured images from webcam
		/// </summary>
		delegate void CaptureCallbackDelegate(
			int dwSize,
			[MarshalAsAttribute(UnmanagedType::LPArray, ArraySubType = UnmanagedType::I1, SizeParamIndex = 0)] array<System::Byte>^ abData);

		/// <summary>
		/// Event callback to capture images from webcam
		/// </summary>
		event CaptureCallbackDelegate^ OnImageCapture;

		/// <summary>
		/// Retrieve information about a specific camera
		/// Use the count property to determine valid indicies to pass in
		/// </summary>
		CameraInfo^ GetCameraInfo(int camIndex);

		/// <summary>
		/// Start the camera associated with the input handle
		/// </summary>
		void StartCamera(int camIndex, interior_ptr<int> width, interior_ptr<int> height);

		/// <summary>
		/// Set VideoProcAmpProperty
		/// </summary>
		void SetProperty(long lProperty, long lValue, bool bAuto);

		/// <summary>
		/// Stops the currently running camera and cleans up any global resources
		/// </summary>
		void Cleanup();

		/// <summary>
		/// Stops the currently running camera
		/// </summary>
		void StopCamera();

		/// <summary>
		/// Show the properties dialog for the specified webcam
		/// </summary>
		void DisplayCameraPropertiesDialog(int camIndex);

		/// <summary>
		/// Count of the number of cameras installed
		/// </summary>
		property int Count;

		/// <summary>
		/// Queries which camera is currently running via StartCamera(), -1 for none
		/// </summary>
		property int ActiveCameraIndex
		{
			int get()
			{
				return activeCameraIndex;
			}
		}

		/// <summary>
		/// IDisposable
		/// </summary>
		~CameraMethods();

	protected:
		/// <summary>
		/// Finalizer
		/// </summary>
		!CameraMethods();

	private:
		/// <summary>
		/// Pinned pointer to delegate for CaptureCallbackDelegate
		/// Keeps the delegate instance in one spot
		/// </summary>
		GCHandle ppCaptureCallback;

		/// <summary>
		/// Initialize information about webcams installed on machine
		/// </summary>
		void RefreshCameraList();

		/// <summary>
		/// Has dispose already happened?
		/// </summary>
		bool disposed;

		/// <summary>
		/// Which camera is running? -1 for none
		/// </summary>
		int activeCameraIndex;

		/// <summary>
		/// Releases all unmanaged resources
		/// </summary>
		void CleanupCameraInfo();

		/// <summary>
		/// Setup the callback functionality for DirectShow
		/// </summary>
		HRESULT ConfigureSampleGrabber(IBaseFilter *pIBaseFilter);

		HRESULT SetCaptureFormat(IBaseFilter* pCap, int width, int height);
	};

	// Forward declarations of callbacks
	typedef void (__stdcall *PFN_CaptureCallback)(DWORD dwSize, BYTE* pbData);
	PFN_CaptureCallback g_pfnCaptureCallback = NULL;

	/// <summary>
	/// Lightweight SampleGrabber callback interface
	/// </summary>
	class SampleGrabberCB : public ISampleGrabberCB
	{
	public:
		SampleGrabberCB()
		{
			m_nRefCount = 0;
		}

		virtual HRESULT STDMETHODCALLTYPE SampleCB(double SampleTime, IMediaSample *pSample)
		{
			return E_FAIL;
		}

		virtual HRESULT STDMETHODCALLTYPE BufferCB(double SampleTime, BYTE *pBuffer, long BufferLen)
		{
			if (g_pfnCaptureCallback != NULL)
			{
				g_pfnCaptureCallback(BufferLen, pBuffer);
			}

			return S_OK;
		}

		virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void **ppvObject)
		{
			return E_FAIL;  // Not a very accurate implementation
		}

		virtual ULONG STDMETHODCALLTYPE AddRef()
		{
			return ++m_nRefCount;
		}

		virtual ULONG STDMETHODCALLTYPE Release()
		{
			int n = --m_nRefCount;
			if (n <= 0)
			{
				delete this;
			}
			return n;
		}

	private:
		int m_nRefCount;
	};
}
