using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace Touchless.Vision.Camera
{
    internal static class WebCamLibInterop
    {

        //[DllImport("WebCamLib.dll", EntryPoint = "Initialize")]
        //public static extern int WebCamInitialize();

        //[DllImport("WebCamLib.dll", EntryPoint = "Cleanup")]
        //public static extern int WebCamCleanup();

        //[DllImport("WebCamLib.dll", EntryPoint = "RefreshCameraList")]
        //public static extern int WebCamRefreshCameraList(ref int count);

        //[DllImport("WebCamLib.dll", EntryPoint = "GetCameraDetails")]
        //public static extern int WebCamGetCameraDetails(int index,
        //    [Out, MarshalAs(UnmanagedType.Interface)] out object nativeInterface,
        //    out IntPtr name);

        public delegate void CaptureCallbackProc(
                int dwSize,
                [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I1, SizeParamIndex = 0)] byte[] abData);

        //[DllImport("WebCamLib.dll", EntryPoint = "StartCamera")]
        //public static extern int WebCamStartCamera(
        //    [In, MarshalAs(UnmanagedType.Interface)] object nativeInterface,
        //    CaptureCallbackProc lpCaptureFunc,
        //    ref int width,
        //    ref int height
        //    );

        //[DllImport("WebCamLib.dll", EntryPoint = "StopCamera")]
        //public static extern int WebCamStopCamera();

        //[DllImport("WebCamLib.dll", EntryPoint = "DisplayCameraPropertiesDialog")]
        //public static extern int WebCamDisplayCameraPropertiesDialog(
        //    [In, MarshalAs(UnmanagedType.Interface)] object nativeInterface);
    }
}
