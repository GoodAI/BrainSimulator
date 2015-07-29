using System;
using System.ComponentModel.Composition;
using Touchless.Vision.Camera.Configuration;
using Touchless.Vision.Contracts;

namespace Touchless.Vision.Camera
{
    [Export(typeof(IFrameSource))]
    public class CameraFrameSource : IFrameSource
    {
        public event Action<IFrameSource, Frame, double> NewFrame;
        
        public bool IsCapturing { get; private set; }

        private Camera _camera;
        public Camera Camera
        {
            get { return _camera; }

            internal set
            {
                if (_camera != value)
                {
                    bool restart = IsCapturing;
                    if (IsCapturing)
                    {
                        StopFrameCapture();
                    }

                    _camera = value;

                    if (restart)
                    {
                        StartFrameCapture();
                    }
                }
            }
        }

        [ImportingConstructor]
        public CameraFrameSource([Import(ExportInterfaceNames.DefaultCamera)] Camera camera)
        {
            if (camera == null) throw new ArgumentNullException("camera");

            this.Camera = camera;
        }

        public void StartFrameCapture()
        {
            if (!IsCapturing && this.Camera != null)
            {
                this.Camera.OnImageCaptured += OnImageCaptured;
                this.Camera.StartCapture();
                IsCapturing = true;
            }
        }

        public void StopFrameCapture()
        {
            if (IsCapturing)
            {
                this.Camera.StopCapture();
                this.Camera.OnImageCaptured += OnImageCaptured;
                this.IsCapturing = false;
            }
        }

        private void OnImageCaptured(object sender, CameraEventArgs e)
        {
            if (IsCapturing && this.NewFrame != null)
            {
                var frame = new Frame(e.Image);
                this.NewFrame(this, frame, e.CameraFps);
            }
        }

        public string Name
        {
            get { return "Touchless Camera Frame Source"; }
        }

        public string Description
        {
            get { return "Retrieves frames from Camera"; }
        }

        public bool HasConfiguration
        {
            get { return true; }
        }


        public System.Windows.UIElement ConfigurationElement
        {
            get
            {
                return new CameraFrameSourceConfigurationElement(this);
            }
        }
    }
}
