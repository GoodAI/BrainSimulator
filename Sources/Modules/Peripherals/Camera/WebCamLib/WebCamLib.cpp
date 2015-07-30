//*****************************************************************************************
//  File:       WebCamLib.cpp
//  Project:    WebcamLib
//  Author(s):  John Conwell
//              Gary Caldwell
//
//  Defines the webcam DirectShow wrapper used by TouchlessLib
//*****************************************************************************************

#include <dshow.h>
#include <strsafe.h>
#define __IDxtCompositor_INTERFACE_DEFINED__
#define __IDxtAlphaSetter_INTERFACE_DEFINED__
#define __IDxtJpeg_INTERFACE_DEFINED__
#define __IDxtKey_INTERFACE_DEFINED__

#pragma include_alias( "dxtrans.h", "qedit.h" )

#include "qedit.h"
#include "WebCamLib.h"

using namespace System;
using namespace System::Reflection;
using namespace WebCamLib;


// Private variables
#define MAX_CAMERAS 10


// Structure to hold camera information
struct CameraInfoStruct
{
	BSTR bstrName;
	IMoniker* pMoniker;
};


// Private global variables
IGraphBuilder* g_pGraphBuilder = NULL;
IMediaControl* g_pMediaControl = NULL;
ICaptureGraphBuilder2* g_pCaptureGraphBuilder = NULL;
IBaseFilter* g_pIBaseFilterCam = NULL;
IBaseFilter* g_pIBaseFilterSampleGrabber = NULL;
IBaseFilter* g_pIBaseFilterNullRenderer = NULL;
CameraInfoStruct g_aCameraInfo[MAX_CAMERAS] = {0};


// http://social.msdn.microsoft.com/Forums/sk/windowsdirectshowdevelopment/thread/052d6a15-f092-4913-b52d-d28f9a51e3b6
void MyFreeMediaType(AM_MEDIA_TYPE& mt) {
    if (mt.cbFormat != 0) {
        CoTaskMemFree((PVOID)mt.pbFormat);
        mt.cbFormat = 0;
        mt.pbFormat = NULL;
    }
    if (mt.pUnk != NULL) {
        // Unecessary because pUnk should not be used, but safest.
        mt.pUnk->Release();
        mt.pUnk = NULL;
    }
}
void MyDeleteMediaType(AM_MEDIA_TYPE *pmt) {
    if (pmt != NULL) {
        MyFreeMediaType(*pmt); // See FreeMediaType for the implementation.
        CoTaskMemFree(pmt);
    }
}


/// <summary>
/// Initializes information about all web cams connected to machine
/// </summary>
CameraMethods::CameraMethods()
{
	// Set to not disposed
	this->disposed = false;

	// Get and cache camera info
	RefreshCameraList();
}

/// <summary>
/// IDispose
/// </summary>
CameraMethods::~CameraMethods()
{
	Cleanup();
	disposed = true;
}

/// <summary>
/// Finalizer
/// </summary>
CameraMethods::!CameraMethods()
{
	if (!disposed)
	{
		Cleanup();
	}
}

/// <summary>
/// Initialize information about webcams installed on machine
/// </summary>
void CameraMethods::RefreshCameraList()
{
	IEnumMoniker* pclassEnum = NULL;
	ICreateDevEnum* pdevEnum = NULL;

	int count = 0;

	CleanupCameraInfo();

	HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum,
		NULL,
		CLSCTX_INPROC,
		IID_ICreateDevEnum,
		(LPVOID*)&pdevEnum);

	if (SUCCEEDED(hr))
	{
		hr = pdevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &pclassEnum, 0);
	}

	if (pdevEnum != NULL)
	{
		pdevEnum->Release();
		pdevEnum = NULL;
	}

	if (pclassEnum != NULL)
	{ 
		IMoniker* apIMoniker[1];
		ULONG ulCount = 0;

		while (SUCCEEDED(hr) && (count) < MAX_CAMERAS && pclassEnum->Next(1, apIMoniker, &ulCount) == S_OK)
		{
			g_aCameraInfo[count].pMoniker = apIMoniker[0];
			g_aCameraInfo[count].pMoniker->AddRef();

			IPropertyBag *pPropBag;
			hr = apIMoniker[0]->BindToStorage(0, 0, IID_IPropertyBag, (void **)&pPropBag);
			if (SUCCEEDED(hr))
			{
				// Retrieve the filter's friendly name
				VARIANT varName;
				VariantInit(&varName);
				hr = pPropBag->Read(L"FriendlyName", &varName, 0);
				if (SUCCEEDED(hr) && varName.vt == VT_BSTR)
				{
					g_aCameraInfo[count].bstrName = SysAllocString(varName.bstrVal);
				}
				VariantClear(&varName);

				pPropBag->Release();
			}

			count++;
		}

		pclassEnum->Release();
	}

	this->Count = count;

	if (!SUCCEEDED(hr))
		throw gcnew COMException("Error Refreshing Camera List", hr);
}

/// <summary>
/// Retrieve information about a specific camera
/// Use the count property to determine valid indicies to pass in
/// </summary>
CameraInfo^ CameraMethods::GetCameraInfo(int camIndex)
{
	if (camIndex >= Count)
		throw gcnew ArgumentException("Camera index is out of bounds: " + Count.ToString());

	if (g_aCameraInfo[camIndex].pMoniker == NULL)
		throw gcnew ArgumentException("There is no camera at index: " + camIndex.ToString());

	CameraInfo^ camInfo = gcnew CameraInfo();
	camInfo->index = camIndex;
	camInfo->name = Marshal::PtrToStringBSTR((IntPtr)g_aCameraInfo[camIndex].bstrName);
	return camInfo;
}

/// <summary>
/// Start the camera associated with the input handle
/// </summary>
void CameraMethods::StartCamera(int camIndex, interior_ptr<int> width, interior_ptr<int> height)
{
	if (camIndex >= Count)
		throw gcnew ArgumentException("Camera index is out of bounds: " + Count.ToString());

	if (g_aCameraInfo[camIndex].pMoniker == NULL)
		throw gcnew ArgumentException("There is no camera at index: " + camIndex.ToString());

	if (g_pGraphBuilder != NULL)
		throw gcnew ArgumentException("Graph Builder was null");

	// Setup up function callback -- through evil reflection on private members
	Type^ baseType = this->GetType();
	FieldInfo^ field = baseType->GetField("<backing_store>OnImageCapture", BindingFlags::NonPublic | BindingFlags::Instance | BindingFlags::IgnoreCase);
	if (field != nullptr)
	{
		Object^ obj = field->GetValue(this);
		if (obj != nullptr)
		{
			CameraMethods::CaptureCallbackDelegate^ del = (CameraMethods::CaptureCallbackDelegate^)field->GetValue(this);
			if (del != nullptr)
			{
				ppCaptureCallback = GCHandle::Alloc(del);
				g_pfnCaptureCallback =
					static_cast<PFN_CaptureCallback>(Marshal::GetFunctionPointerForDelegate(del).ToPointer());
			}
		}
	}

	IMoniker *pMoniker = g_aCameraInfo[camIndex].pMoniker;
	pMoniker->AddRef();

	HRESULT hr = S_OK;

	// Build all the necessary interfaces to start the capture
	if (SUCCEEDED(hr))
	{
		hr = CoCreateInstance(CLSID_FilterGraph,
			NULL,
			CLSCTX_INPROC,
			IID_IGraphBuilder,
			(LPVOID*)&g_pGraphBuilder);
	}

	if (SUCCEEDED(hr))
	{
		hr = g_pGraphBuilder->QueryInterface(IID_IMediaControl, (LPVOID*)&g_pMediaControl);
	}

	if (SUCCEEDED(hr))
	{
		hr = CoCreateInstance(CLSID_CaptureGraphBuilder2,
			NULL,
			CLSCTX_INPROC,
			IID_ICaptureGraphBuilder2,
			(LPVOID*)&g_pCaptureGraphBuilder);
	}

	// Setup the filter graph
	if (SUCCEEDED(hr))
	{
		hr = g_pCaptureGraphBuilder->SetFiltergraph(g_pGraphBuilder);
	}

	// Build the camera from the moniker
	if (SUCCEEDED(hr))
	{
		hr = pMoniker->BindToObject(NULL, NULL, IID_IBaseFilter, (LPVOID*)&g_pIBaseFilterCam);
	}

	// Add the camera to the filter graph
	if (SUCCEEDED(hr))
	{
		hr = g_pGraphBuilder->AddFilter(g_pIBaseFilterCam, L"WebCam");
	}

	// Set the resolution
    if (SUCCEEDED(hr)) {
		hr = SetCaptureFormat(g_pIBaseFilterCam, *width, *height);
	}

	// Create a SampleGrabber
	if (SUCCEEDED(hr))
	{
		hr = CoCreateInstance(CLSID_SampleGrabber, NULL, CLSCTX_INPROC_SERVER, IID_IBaseFilter, (void**)&g_pIBaseFilterSampleGrabber);
	}

	// Configure the Sample Grabber
	if (SUCCEEDED(hr))
	{
		hr = ConfigureSampleGrabber(g_pIBaseFilterSampleGrabber);
	}

	
	// Add Sample Grabber to the filter graph
	if (SUCCEEDED(hr))
	{
		hr = g_pGraphBuilder->AddFilter(g_pIBaseFilterSampleGrabber, L"SampleGrabber");
	}

	// Create the NullRender
	if (SUCCEEDED(hr))
	{
		hr = CoCreateInstance(CLSID_NullRenderer, NULL, CLSCTX_INPROC_SERVER, IID_IBaseFilter, (void**)&g_pIBaseFilterNullRenderer);
	}

	// Add the Null Render to the filter graph
	if (SUCCEEDED(hr))
	{
		hr = g_pGraphBuilder->AddFilter(g_pIBaseFilterNullRenderer, L"NullRenderer");
	}

	// Configure the render stream
	if (SUCCEEDED(hr))
	{
		hr = g_pCaptureGraphBuilder->RenderStream(&PIN_CATEGORY_CAPTURE, &MEDIATYPE_Video, g_pIBaseFilterCam, g_pIBaseFilterSampleGrabber, g_pIBaseFilterNullRenderer);
	}

	// Grab the capture width and height
	if (SUCCEEDED(hr))
	{
		ISampleGrabber* pGrabber = NULL;
		hr = g_pIBaseFilterSampleGrabber->QueryInterface(IID_ISampleGrabber, (LPVOID*)&pGrabber);
		if (SUCCEEDED(hr))
		{
			AM_MEDIA_TYPE mt;
			hr = pGrabber->GetConnectedMediaType(&mt);
			if (SUCCEEDED(hr))
			{
				VIDEOINFOHEADER *pVih;
				if ((mt.formattype == FORMAT_VideoInfo) &&
					(mt.cbFormat >= sizeof(VIDEOINFOHEADER)) &&
					(mt.pbFormat != NULL) )
				{
					pVih = (VIDEOINFOHEADER*)mt.pbFormat;
					*width = pVih->bmiHeader.biWidth;
					*height = pVih->bmiHeader.biHeight;
				}
				else
				{
					hr = E_FAIL;  // Wrong format
				}

				// FreeMediaType(mt); (from MSDN)
				if (mt.cbFormat != 0)
				{
					CoTaskMemFree((PVOID)mt.pbFormat);
					mt.cbFormat = 0;
					mt.pbFormat = NULL;
				}
				if (mt.pUnk != NULL)
				{
					// Unecessary because pUnk should not be used, but safest.
					mt.pUnk->Release();
					mt.pUnk = NULL;
				}
			}
		}

		if (pGrabber != NULL)
		{
			pGrabber->Release();
			pGrabber = NULL;
		}
	}

	// Start the capture
	if (SUCCEEDED(hr))
	{
		hr = g_pMediaControl->Run();
	}

	// If init fails then ensure that you cleanup
	if (FAILED(hr))
	{
		StopCamera();
	}
	else
	{
		hr = S_OK;  // Make sure we return S_OK for success
	}

	// Cleanup
	if (pMoniker != NULL)
	{
		pMoniker->Release();
		pMoniker = NULL;
	}

	if (SUCCEEDED(hr))
		this->activeCameraIndex = camIndex;
	else
		throw gcnew COMException("Error Starting Camera", hr);
}

void CameraMethods::SetProperty(long lProperty, long lValue, bool bAuto)
{
	if (g_pIBaseFilterCam == NULL) throw gcnew ArgumentException("No Camera started"); 

    HRESULT hr = S_OK;

	// Query the capture filter for the IAMVideoProcAmp interface.
	IAMVideoProcAmp *pProcAmp = 0;
	hr = g_pIBaseFilterCam->QueryInterface(IID_IAMVideoProcAmp, (void**)&pProcAmp);

	// Get the range and default value.
    long Min, Max, Step, Default, Flags;
	if (SUCCEEDED(hr)) {
	   hr = pProcAmp->GetRange(lProperty, &Min, &Max, &Step, &Default, &Flags);
	}

	if (SUCCEEDED(hr)) {
		lValue = Min + (Max - Min) * lValue / 100;
		hr = pProcAmp->Set(lProperty, lValue, bAuto ? VideoProcAmp_Flags_Auto : VideoProcAmp_Flags_Manual);
	}

	if (!SUCCEEDED(hr)) throw gcnew COMException("Error Set Property", hr);
}

/// <summary>
/// Closes any open webcam and releases all unmanaged resources
/// </summary>
void CameraMethods::Cleanup()
{
	StopCamera();
	CleanupCameraInfo();

	// Clean up pinned pointer to callback delegate
	if (ppCaptureCallback.IsAllocated)
	{
		ppCaptureCallback.Free();
	}
}

/// <summary>
/// Stops the current open webcam
/// </summary>
void CameraMethods::StopCamera()
{
	if (g_pMediaControl != NULL)
	{
		g_pMediaControl->Stop();
		g_pMediaControl->Release();
		g_pMediaControl = NULL;
	}

	g_pfnCaptureCallback = NULL;

	if (g_pIBaseFilterNullRenderer != NULL)
	{
		g_pIBaseFilterNullRenderer->Release();
		g_pIBaseFilterNullRenderer = NULL;
	}

	if (g_pIBaseFilterSampleGrabber != NULL)
	{
		g_pIBaseFilterSampleGrabber->Release();
		g_pIBaseFilterSampleGrabber = NULL;
	}

	if (g_pIBaseFilterCam != NULL)
	{
		g_pIBaseFilterCam->Release();
		g_pIBaseFilterCam = NULL;
	}

	if (g_pGraphBuilder != NULL)
	{
		g_pGraphBuilder->Release();
		g_pGraphBuilder = NULL;
	}

	if (g_pCaptureGraphBuilder != NULL)
	{
		g_pCaptureGraphBuilder->Release();
		g_pCaptureGraphBuilder = NULL;
	}

	this->activeCameraIndex = -1;
}

/// <summary>
/// Show the properties dialog for the specified webcam
/// </summary>
void CameraMethods::DisplayCameraPropertiesDialog(int camIndex)
{
	if (camIndex >= Count)
		throw gcnew ArgumentException("Camera index is out of bounds: " + Count.ToString());

	if (g_aCameraInfo[camIndex].pMoniker == NULL)
		throw gcnew ArgumentException("There is no camera at index: " + camIndex.ToString());

	HRESULT hr = S_OK;
	IBaseFilter *pFilter = NULL;
	ISpecifyPropertyPages *pProp = NULL;
	IMoniker *pMoniker = g_aCameraInfo[camIndex].pMoniker;
	pMoniker->AddRef();

	// Create a filter graph for the moniker
	if (SUCCEEDED(hr))
	{
		hr = pMoniker->BindToObject(NULL, NULL, IID_IBaseFilter, (LPVOID*)&pFilter);
	}

	// See if it implements a property page
	if (SUCCEEDED(hr))
	{
		hr = pFilter->QueryInterface(IID_ISpecifyPropertyPages, (LPVOID*)&pProp);
	}

	// Show the property page
	if (SUCCEEDED(hr))
	{
		FILTER_INFO filterinfo;
		hr = pFilter->QueryFilterInfo(&filterinfo);

		IUnknown *pFilterUnk = NULL;
		if (SUCCEEDED(hr))
		{
			hr = pFilter->QueryInterface(IID_IUnknown, (LPVOID*)&pFilterUnk);
		}

		if (SUCCEEDED(hr))
		{
			CAUUID caGUID;
			pProp->GetPages(&caGUID);

			OleCreatePropertyFrame(
				NULL,                   // Parent window
				0, 0,                   // Reserved
				filterinfo.achName,     // Caption for the dialog box
				1,                      // Number of objects (just the filter)
				&pFilterUnk,            // Array of object pointers. 
				caGUID.cElems,          // Number of property pages
				caGUID.pElems,          // Array of property page CLSIDs
				0,                      // Locale identifier
				0, NULL                 // Reserved
				);
		}

		if (pFilterUnk != NULL)
		{
			pFilterUnk->Release();
			pFilterUnk = NULL;
		}
	}

	if (pProp != NULL)
	{
		pProp->Release();
		pProp = NULL;
	}

	if (pMoniker != NULL)
	{
		pMoniker->Release();
		pMoniker = NULL;
	}

	if (pFilter != NULL)
	{
		pFilter->Release();
		pFilter = NULL;
	}

	if (!SUCCEEDED(hr))
		throw gcnew COMException("Error displaying camera properties dialog", hr);
}

/// <summary>
/// Releases all unmanaged resources
/// </summary>
void CameraMethods::CleanupCameraInfo()
{
	for (int n = 0; n < MAX_CAMERAS; n++)
	{
		SysFreeString(g_aCameraInfo[n].bstrName);
		g_aCameraInfo[n].bstrName = NULL;
		if (g_aCameraInfo[n].pMoniker != NULL)
		{
			g_aCameraInfo[n].pMoniker->Release();
			g_aCameraInfo[n].pMoniker = NULL;
		}
	}
}


/// <summary>
/// Setup the callback functionality for DirectShow
/// </summary>
HRESULT CameraMethods::ConfigureSampleGrabber(IBaseFilter *pIBaseFilter)
{
	HRESULT hr = S_OK;

	ISampleGrabber *pGrabber = NULL;

	hr = pIBaseFilter->QueryInterface(IID_ISampleGrabber, (void**)&pGrabber);
	if (SUCCEEDED(hr))
	{
		AM_MEDIA_TYPE mt;
		ZeroMemory(&mt, sizeof(AM_MEDIA_TYPE));
		mt.majortype = MEDIATYPE_Video;
		mt.subtype = MEDIASUBTYPE_RGB24;
		mt.formattype = FORMAT_VideoInfo;
		hr = pGrabber->SetMediaType(&mt);
	}

	if (SUCCEEDED(hr))
	{
		hr = pGrabber->SetCallback(new SampleGrabberCB(), 1);
	}

	if (pGrabber != NULL)
	{
		pGrabber->Release();
		pGrabber = NULL;
	}

	return hr;
}



// based on http://stackoverflow.com/questions/7383372/cant-make-iamstreamconfig-setformat-to-work-with-lifecam-studio
HRESULT CameraMethods::SetCaptureFormat(IBaseFilter* pCap, int width, int height)
{
    HRESULT hr = S_OK;

    IAMStreamConfig *pConfig = NULL;
    hr = g_pCaptureGraphBuilder->FindInterface(
        &PIN_CATEGORY_CAPTURE,
		&MEDIATYPE_Video, 
        pCap, // Pointer to the capture filter.
        IID_IAMStreamConfig, (void**)&pConfig);
    if (!SUCCEEDED(hr)) return hr;

    int iCount = 0, iSize = 0;
    hr = pConfig->GetNumberOfCapabilities(&iCount, &iSize);
	if (!SUCCEEDED(hr)) return hr;

    // Check the size to make sure we pass in the correct structure.
    if (iSize == sizeof(VIDEO_STREAM_CONFIG_CAPS))
	{
        // Use the video capabilities structure.
        for (int iFormat = 0; iFormat < iCount; iFormat++)
        {
            VIDEO_STREAM_CONFIG_CAPS scc;
            AM_MEDIA_TYPE *pmt;
            /* Note:  Use of the VIDEO_STREAM_CONFIG_CAPS structure to configure a video device is 
            deprecated. Although the caller must allocate the buffer, it should ignore the 
            contents after the method returns. The capture device will return its supported 
            formats through the pmt parameter. */
            hr = pConfig->GetStreamCaps(iFormat, &pmt, (BYTE*)&scc);
            if (SUCCEEDED(hr))
            {
                /* Examine the format, and possibly use it. */
                if (pmt->formattype == FORMAT_VideoInfo) {
                    // Check the buffer size.
                    if (pmt->cbFormat >= sizeof(VIDEOINFOHEADER))
                    {
                        VIDEOINFOHEADER *pVih =  reinterpret_cast<VIDEOINFOHEADER*>(pmt->pbFormat);
                        BITMAPINFOHEADER *bmiHeader = &pVih->bmiHeader;

                        /* Access VIDEOINFOHEADER members through pVih. */
                        if( bmiHeader->biWidth == width && bmiHeader->biHeight == height && 
                            bmiHeader->biBitCount == 24)
                        {
                            hr = pConfig->SetFormat(pmt);

							MyDeleteMediaType(pmt); break;
                        }
                    }
                }

                // Delete the media type when you are done.
                MyDeleteMediaType(pmt);
            }
        }
    }

	return hr;
}
