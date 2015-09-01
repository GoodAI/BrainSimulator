using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using System.Windows.Forms;

namespace GoodAI.BrainSimulator.Utils
{    
    public abstract class MyCamera 
    {
        protected GLControl m_glControl;
        protected Vector2 m_mouseStartDrag;
        protected Vector2 m_mousePos;
        protected Vector2 m_mouseLastPos;
        protected MouseButtons m_mouseState = 0;

        protected const float ZOOM_SENSITIVITY = 0.005f;

        public abstract MyCameraData CameraSetup { get; set; }        

        public MyCamera(GLControl glControl)
        {
            m_glControl = glControl;

            glControl.MouseDown += new MouseEventHandler(glControl_MouseDown);
            glControl.MouseMove += new MouseEventHandler(glControl_MouseMove);
            glControl.MouseUp += new MouseEventHandler(glControl_MouseUp);
        }

        protected abstract void ProcessMouseMovement();        
        public abstract void ApplyCameraTransform();        

        private void glControl_MouseDown(object sender, MouseEventArgs e)
        {
            m_mouseState |= e.Button;
            m_mouseStartDrag.X = e.Location.X;
            m_mouseStartDrag.Y = e.Location.Y;
        }

        private void glControl_MouseUp(object sender, MouseEventArgs e)
        {
            m_mouseState &= ~e.Button;
        }

        private void glControl_MouseMove(object sender, MouseEventArgs e)
        {
            m_mousePos.X = e.Location.X;
            m_mousePos.Y = e.Location.Y;

            ProcessMouseMovement();

            m_mouseLastPos = m_mousePos;                        
        }
    }

    public class MyZoomAndPanCamera : MyCamera
    {
        private Vector3 m_zoomAndPan;

        private const float PAN_SENSITIVITY = 0.0012f;

        public MyZoomAndPanCamera(GLControl glControl) : base(glControl)
        {           
            m_zoomAndPan.Z = 5.0f;
        }

        protected override void ProcessMouseMovement()
        {
            if ((m_mouseState & MouseButtons.Middle) > 0)
            {
                Vector2 mouseDelta = m_mouseLastPos - m_mousePos;
                m_zoomAndPan.Z *= 1 + (mouseDelta.X + mouseDelta.Y) * ZOOM_SENSITIVITY;

                m_glControl.Invalidate();
            }

            if ((m_mouseState & MouseButtons.Right) > 0)
            {
                Vector2 mouseDelta = m_mouseLastPos - m_mousePos;
                m_zoomAndPan.X += mouseDelta.X * PAN_SENSITIVITY * m_zoomAndPan.Z;
                m_zoomAndPan.Y += -mouseDelta.Y * PAN_SENSITIVITY * m_zoomAndPan.Z;

                m_glControl.Invalidate();
            }
        }

        public override void ApplyCameraTransform()
        {
            GL.Translate(-m_zoomAndPan);   
        }

        public override MyCameraData CameraSetup
        {
            get
            {
                MyCameraData result = new MyCameraData()
                {
                    CameraType = MyAbstractObserver.ViewMethod.Free_2D,
                    X = m_zoomAndPan.X,
                    Y = m_zoomAndPan.Y,
                    Z = m_zoomAndPan.Z
                };
                return result;
            }
            set
            {
                m_zoomAndPan.X = value.X;
                m_zoomAndPan.Y = value.Y;
                m_zoomAndPan.Z = value.Z;
            }
        }
    }

    public class MyOrbitCamera : MyCamera
    {
        private float m_radius;
        private float m_azimuth;
        private float m_inclination;
        
        private const float ORBIT_SENSITIVITY = 0.75f;

        public MyOrbitCamera(GLControl glControl) : base(glControl)
        {           
            m_radius = 5.0f;
            m_inclination = 30.0f;
            m_azimuth = -45.0f;
        }

        protected override void ProcessMouseMovement()
        {            
            if ((m_mouseState & MouseButtons.Middle) > 0)
            {
                Vector2 mouseDelta = m_mouseLastPos - m_mousePos;
                m_radius *= 1 + (mouseDelta.X + mouseDelta.Y) * ZOOM_SENSITIVITY;

                m_glControl.Invalidate();
            }

            if ((m_mouseState & MouseButtons.Right) > 0)
            {
                Vector2 mouseDelta = m_mouseLastPos - m_mousePos;
                m_azimuth += -mouseDelta.X * ORBIT_SENSITIVITY;
                m_inclination += -mouseDelta.Y * ORBIT_SENSITIVITY;

                m_glControl.Invalidate();
            }
        }

        public override void ApplyCameraTransform()
        {
            GL.Translate(0, 0, -m_radius);
            GL.Rotate(m_inclination, 1, 0, 0);
            GL.Rotate(m_azimuth, 0, 1, 0);            
        }

        public override MyCameraData CameraSetup
        {
            get
            {
                MyCameraData result = new MyCameraData()
                {
                    CameraType = MyAbstractObserver.ViewMethod.Orbit_3D,
                    X = m_azimuth,
                    Y = m_inclination,
                    Z = m_radius,                    
                };
                return result;
            }
            set
            {
                m_azimuth = value.X;
                m_inclination = value.Y;
                m_radius = value.Z;
            }
        }
    }
}
