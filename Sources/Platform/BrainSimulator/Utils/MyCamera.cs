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
        protected const float PAN_SENSITIVITY = 0.0012f;

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
        private float m_azimuth;
        private float m_inclination;
        private Vector3 m_translation;
        
        private const float ORBIT_SENSITIVITY = 0.5f;

        public MyOrbitCamera(GLControl glControl) : base(glControl)
        {           
            m_azimuth = -45.0f;
            m_inclination = 30.0f;
            m_translation = new Vector3(0, 0, -5);
        }

        protected override void ProcessMouseMovement()
        {            
            Vector2 mouseDelta = m_mouseLastPos - m_mousePos;

            if ((m_mouseState & MouseButtons.Left) > 0)
            {
                m_translation.X += mouseDelta.X * PAN_SENSITIVITY * m_translation.Z;
                m_translation.Y -= mouseDelta.Y * PAN_SENSITIVITY * m_translation.Z;

                m_glControl.Invalidate();
            }

            if ((m_mouseState & MouseButtons.Middle) > 0)
            {
                m_translation.Z *= 1 + (mouseDelta.X + mouseDelta.Y) * ZOOM_SENSITIVITY; 

                m_glControl.Invalidate();
            }

            if ((m_mouseState & MouseButtons.Right) > 0)
            {
                m_azimuth += -mouseDelta.X * ORBIT_SENSITIVITY;
                m_inclination += -mouseDelta.Y * ORBIT_SENSITIVITY;

                m_glControl.Invalidate();
            }
        }

        public override void ApplyCameraTransform()
        {
            GL.Translate(m_translation);
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
                    Azimuth = m_azimuth,
                    Inclination = m_inclination,
                    X = m_translation.X,                    
                    Y = m_translation.Y,                    
                    Z = -m_translation.Z,                    
                };
                return result;
            }
            set
            {
                m_azimuth = value.Azimuth;
                m_inclination = value.Inclination;
                m_translation.X = value.X;
                m_translation.Y = value.Y;
                m_translation.Z = -value.Z;
            }
        }
    }
}
