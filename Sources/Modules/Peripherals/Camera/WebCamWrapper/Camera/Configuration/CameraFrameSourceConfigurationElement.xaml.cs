using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.ComponentModel;
using Touchless.Shared.Extensions;

namespace Touchless.Vision.Camera.Configuration
{
    /// <summary>
    /// Interaction logic for CameraFrameSourceConfigurationElement.xaml
    /// </summary>
    public partial class CameraFrameSourceConfigurationElement : UserControl
    {
        private readonly CameraFrameSource _cameraFrameSource;

        public CameraFrameSourceConfigurationElement()
            : this(null)
        {

        }

        public CameraFrameSourceConfigurationElement(CameraFrameSource cameraFrameSource)
        {
            _cameraFrameSource = cameraFrameSource;
            InitializeComponent();

            if (!DesignerProperties.GetIsInDesignMode(this) && _cameraFrameSource != null)
            {
                _cameraFrameSource.NewFrame += NewFrame;

                var cameras = CameraService.AvailableCameras.ToList();
                comboBoxCameras.ItemsSource = cameras;
                comboBoxCameras.SelectedItem = _cameraFrameSource.Camera;
            }
        }

        private readonly object _frameSync = new object();
        private bool _frameWaiting = false;
        private void NewFrame(Touchless.Vision.Contracts.IFrameSource frameSource, Touchless.Vision.Contracts.Frame frame, double fps)
        {
            //We want to ignore frames we can't render fast enough
            lock (_frameSync)
            {
                if (!_frameWaiting)
                {
                    _frameWaiting = true;
                    Action workAction = delegate
                    {
                        this.labelCameraFPSValue.Content = fps.ToString();
                        this.imgPreview.Source = frame.OriginalImage.ToBitmapSource();

                        lock (_frameSync)
                        {
                            _frameWaiting = false;
                        }
                    };
                    Dispatcher.BeginInvoke(workAction);
                }
            }
        }

        private void comboBoxCameras_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            _cameraFrameSource.Camera = comboBoxCameras.SelectedItem as Camera;
            panelCameraInfo.Visibility = comboBoxCameras.SelectedItem != null
                                                ? Visibility.Visible
                                                : Visibility.Collapsed;
        }

        private void buttonCameraProperties_Click(object sender, RoutedEventArgs e)
        {
            _cameraFrameSource.Camera.ShowPropertiesDialog();
        }

        private void chkLimitFps_Checked(object sender, RoutedEventArgs e)
        {
            this.txtLimitFps.IsEnabled = true;
            updateFPS();
        }

        private void chkLimitFps_Unchecked(object sender, RoutedEventArgs e)
        {
            //this.txtLimitFps.Text = "-1";
            this.txtLimitFps.Background = Brushes.White;
            _cameraFrameSource.Camera.Fps = -1;
            this.txtLimitFps.IsEnabled = false;
        }

        private void txtLimitFps_TextChanged(object sender, TextChangedEventArgs e)
        {
            updateFPS();
        }

        private void updateFPS()
        {
            int fps;
            if (int.TryParse(this.txtLimitFps.Text, out fps))
            {
                _cameraFrameSource.Camera.Fps = fps;
                this.txtLimitFps.Background = Brushes.LightGreen;
            }
            else
            {
                this.txtLimitFps.Background = Brushes.Red;
            }
        }
    }
}
