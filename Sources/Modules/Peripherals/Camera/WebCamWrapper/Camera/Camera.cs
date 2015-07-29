using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using WebCamLib;

namespace Touchless.Vision.Camera
{
    /// <summary>
    /// Represents a camera in use by the Touchless system
    /// </summary>
    public class Camera : IDisposable
    {
        private readonly CameraMethods _cameraMethods;
        private RotateFlipType _rotateFlip = RotateFlipType.RotateNoneFlipNone;

        public Camera(CameraMethods cameraMethods, string name, int index)
        {
            _cameraMethods = cameraMethods;
            _name = name;
            _index = index;
            _cameraMethods.OnImageCapture += CaptureCallbackProc;
        }

        public string Name
        {
            get { return _name; }
        }

        /// <summary>
        /// Defines the frames per second limit that is in place, -1 means no limit
        /// </summary>
        public int Fps
        {
            get { return _fpslimit; }
            set
            {
                _fpslimit = value;
                _timeBetweenFrames = (1000.0 / _fpslimit);
            }
        }

        /// <summary>
        /// Determines the width of the image captured
        /// </summary>
        public int CaptureWidth
        {
            get
            {
                return _width;
            }
            set
            {
                _width = value;
            }
        }

        /// <summary>
        /// Defines the height of the image captured
        /// </summary>
        public int CaptureHeight
        {
            get
            {
                return _height;
            }
            set
            {
                _height = value;
            }
        }

        public bool HasFrameLimit
        {
            get { return _fpslimit != -1; }
        }

        public bool FlipHorizontal
        {
            get
            {
                return RotateFlip == RotateFlipType.RotateNoneFlipX || RotateFlip == RotateFlipType.Rotate180FlipNone;
            }

            set
            {
                if (value && FlipVertical)
                {
                    RotateFlip = RotateFlipType.Rotate180FlipNone;
                }
                else if (value && !FlipVertical)
                {
                    RotateFlip = RotateFlipType.RotateNoneFlipX;
                }
                else if (!value && FlipVertical)
                {
                    RotateFlip = RotateFlipType.Rotate180FlipX;
                }
                else if (!value && !FlipVertical)
                {
                    RotateFlip = RotateFlipType.RotateNoneFlipNone;
                }
            }
        }

        public bool FlipVertical
        {
            get
            {
                return RotateFlip == RotateFlipType.Rotate180FlipY || RotateFlip == RotateFlipType.Rotate180FlipNone;
            }

            set
            {
                if (value && FlipVertical)
                {
                    RotateFlip = RotateFlipType.Rotate180FlipNone;
                }
                else if (value && !FlipVertical)
                {
                    RotateFlip = RotateFlipType.Rotate180FlipY;
                }
                else if (!value && FlipVertical)
                {
                    RotateFlip = RotateFlipType.RotateNoneFlipY;
                }
                else if (!value && !FlipVertical)
                {
                    RotateFlip = RotateFlipType.RotateNoneFlipNone;
                }
            }
        }

        /// <summary>
        /// Command for rotating and flipping incoming images
        /// </summary>
        public RotateFlipType RotateFlip
        {
            get { return _rotateFlip; }
            set
            {
                // Swap height/width when rotating by 90 or 270
                if ((int)_rotateFlip % 2 != (int)value % 2)
                {
                    int temp = CaptureWidth;
                    CaptureWidth = CaptureHeight;
                    CaptureHeight = temp;
                }
                _rotateFlip = value;
            }
        }

        #region IDisposable Members

        /// <summary>
        /// Cleanup function for the camera
        /// </summary>
        public void Dispose()
        {
            StopCapture();
        }

        #endregion

        /// <summary>
        /// Returns the last image acquired from the camera
        /// </summary>
        /// <returns>A bitmap of the last image acquired from the camera</returns>
        public Bitmap GetCurrentImage()
        {
            Bitmap b = null;
            lock (_bitmapLock)
            {
                if (_bitmap == null)
                {
                    return null;
                }

                b = (Bitmap)_bitmap.Clone();
            }

            return b;
        }

        public void ShowPropertiesDialog()
        {
            _cameraMethods.DisplayCameraPropertiesDialog(_index); 
        }

        public void GetCameraInfo()
        {
            _cameraMethods.GetCameraInfo(_index);
        }

        /// <summary>
        /// Event fired when an image from the camera is captured
        /// </summary>
        public event EventHandler<CameraEventArgs> OnImageCaptured;

        /// <summary>
        /// Returns the camera name as the ToString implementation
        /// </summary>
        /// <returns>The name of the camera</returns>
        public override string ToString()
        {
            return _name;
        }

        #region Internal Implementation

        private readonly object _bitmapLock = new object();
        private readonly int _index;
        private readonly string _name;
        private Bitmap _bitmap;
        private DateTime _dtLastCap = DateTime.MinValue;
        private int _fpslimit = -1;
        private int _height = 240;
        private double _timeBehind;
        private double _timeBetweenFrames;
        private int _width = 320;

        internal void StartCapture()
        {
            _cameraMethods.StartCamera(_index, ref _width, ref _height);
        }

        internal void StopCapture()
        {
            _cameraMethods.StopCamera();
        }

        /// <summary>
        /// Here is where the images come in as they are collected, as fast as they can and on a background thread
        /// </summary>
private void CaptureCallbackProc(int dataSize, byte[] data)
{
    // Do the magic to create a bitmap
    int stride = _width * 3;
    GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
    var scan0 = (int)handle.AddrOfPinnedObject();
    scan0 += (_height - 1) * stride;
    var b = new Bitmap(_width, _height, -stride, PixelFormat.Format24bppRgb, (IntPtr)scan0);
    b.RotateFlip(_rotateFlip);
    // Copy the image using the Thumbnail function to also resize if needed
    var copyBitmap = (Bitmap)b.GetThumbnailImage(_width, _height, null, IntPtr.Zero);
    //var copyBitmap = (Bitmap)b.Clone();

    // Now you can free the handle
    handle.Free();

    ImageCaptured(copyBitmap);
}

private void ImageCaptured(Bitmap bitmap)
{
    DateTime dtCap = DateTime.Now;

    // Always save the bitmap
    lock (_bitmapLock)
    {
        _bitmap = bitmap;
    }

    // FPS affects the callbacks only
    if (_fpslimit != -1)
    {
        if (_dtLastCap != DateTime.MinValue)
        {
            double milliseconds = ((dtCap.Ticks - _dtLastCap.Ticks) / TimeSpan.TicksPerMillisecond) * 1.15;
            if (milliseconds + _timeBehind >= _timeBetweenFrames)
            {
                _timeBehind = (milliseconds - _timeBetweenFrames);
                if (_timeBehind < 0.0)
                {
                    _timeBehind = 0.0;
                }
            }
            else
            {
                _timeBehind = 0.0;
                return; // ignore the frame
            }
        }
    }

    if (OnImageCaptured != null)
    {
        var fps = (int)(1 / dtCap.Subtract(_dtLastCap).TotalSeconds);
        OnImageCaptured.Invoke(this, new CameraEventArgs(bitmap, fps));
    }

    _dtLastCap = dtCap;
}

        #endregion
    }

    /// <summary>
    /// Camera specific EventArgs that provides the Image being captured
    /// </summary>
    public class CameraEventArgs : EventArgs
    {
        /// <summary>
        /// Current Camera Image
        /// </summary>
        public Bitmap Image
        {
            get { return _image; }
        }

        public int CameraFps
        {
            get { return _cameraFps; }
        }

        #region Internal Implementation

        private readonly int _cameraFps;
        private readonly Bitmap _image;

        internal CameraEventArgs(Bitmap i, int fps)
        {
            _image = i;
            _cameraFps = fps;
        }

        #endregion
    }
}