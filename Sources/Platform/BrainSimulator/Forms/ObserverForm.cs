using GoodAI.BrainSimulator.Utils;
using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using ManagedCuda.VectorTypes;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.BrainSimulator.Forms
{
    public partial class ObserverForm : DockContent
    {
        protected MainForm m_mainForm;
        protected bool m_initialized;        

        public bool Active { get; set; }        

        public MyAbstractObserver Observer { get; private set; }

        private MyCamera m_camera;
        
        public ObserverForm(MainForm mainForm, MyAbstractObserver observer, MyNode declaredOwner) 
        {
            InitializeComponent();
            m_mainForm = mainForm;

            Observer = observer;            
            observer.TriggerReset();

            Text = observer.GetTargetName(declaredOwner);
        }

        protected virtual void ObserverForm_Load(object sender, EventArgs e)
        {
            glControl.MakeCurrent();  
            GL.ClearColor(Color.LightGray);

            GL.Enable(EnableCap.Normalize);            
            GL.Enable(EnableCap.Lighting);
            GL.Enable(EnableCap.Light0);
            GL.Enable(EnableCap.ColorMaterial);
            GL.Light(LightName.Light0, LightParameter.Diffuse, new Color4(1.0f, 1.0f, 1.0f, 1.0f));
            GL.Light(LightName.Light0, LightParameter.Ambient, new Color4(0.1f, 0.1f, 0.1f, 1.0f));                           

            m_mainForm.SimulationHandler.StateChanged += SimulationHandler_StateChanged;
            m_mainForm.SimulationHandler.ProgressChanged += SimulationHandler_ProgressChanged;

            m_initialized = true;
            Observer.Initialized = true;

            this.Activated += ObserverForm_Activated;
        }

        void ObserverForm_Activated(object sender, EventArgs e)
        {
            FocusWindow();
        }

        public void FocusWindow()
        {
            foreach (GraphLayoutForm graphView in m_mainForm.GraphViews.Values)
            {
                graphView.Desktop.FocusElement = null;
            }
            m_mainForm.NodePropertyView.Target = Observer;
        }

        public void StoreWindowInfo()
        {            
            Observer.WindowLocation = new MyLocation()
            {
                X = FloatPane.FloatWindow.Location.X,
                Y = FloatPane.FloatWindow.Location.Y
            };

            Observer.WindowSize = new MySize()
            {
                Width = FloatPane.FloatWindow.Size.Width,
                Height = FloatPane.FloatWindow.Size.Height
            };

            if (m_camera != null)
            {
                Observer.CameraData = m_camera.CameraSetup;                
            }
        }        

        protected virtual void RenderFrame()
        {
            if (!m_initialized) return;
                       
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            GL.Enable(EnableCap.DepthTest);
            GL.Disable(EnableCap.Lighting);
            GL.Disable(EnableCap.Blend);

            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadIdentity();              

            if (Active)
            {
                if (Observer.ViewMode != MyAbstractObserver.ViewMethod.Fit_2D)
                {
                    m_camera.ApplyCameraTransform();

                    GL.Light(LightName.Light0, LightParameter.Position, new Vector4(0.3f, 1.0f, 0.5f, 0));          

                    GL.Disable(EnableCap.Texture2D);

                    GL.PushMatrix();
                    GL.Translate(0, 0, 0.001);
                    MyRenderUtils.RenderAxes(Observer.ViewMode == MyAbstractObserver.ViewMethod.Orbit_3D);
                    GL.Translate(0, -0.001, 0);
                    MyRenderUtils.RenderGrid(Observer.ViewMode == MyAbstractObserver.ViewMethod.Orbit_3D);
                    GL.PopMatrix();
                }

                for (int i = 0; i < Observer.Shapes.Count; i++)
                {
                    GL.PushMatrix();
                    GL.Translate(Observer.Shapes[i].Translation);
                    Observer.Shapes[i].Render();
                    GL.PopMatrix();
                }
            }
            else
            {
                GL.Disable(EnableCap.Texture2D);
                GL.Color3(1.0f, 0, 0);

                GL.Begin(PrimitiveType.Lines);
                GL.Vertex2(-1.0, 1.0);
                GL.Vertex2(1.0, -1.0);

                GL.Vertex2(1.0, 1.0);
                GL.Vertex2(-1.0, -1.0);
                GL.End();
            }

            glControl.SwapBuffers();
        }

        internal void UpdateView(uint simulationStep)
        {
            if (!this.IsDisposed)
            {
                peekLabel.Visible = false;

                try
                {
                    Observer.UpdateFrame(simulationStep);
                }
                catch (Exception exc)
                {
                    MyLog.ERROR.WriteLine("Observer update failed: " + exc.Message);
                }
                finally
                {
                    glControl.Invalidate();
                }

                if (Observer.AutosaveSnapshop)
                {
                    snapshotToolStripMenuItem_Click(this, EventArgs.Empty);
                }
            }
        }

        protected virtual void SetupViewport()
        {            
            GL.MatrixMode(MatrixMode.Projection);
            GL.LoadIdentity();            

            if (!Observer.Active)
            {
                GL.Ortho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
            }
            else
            {
                double windowRatio = (double)glControl.Width / glControl.Height;
                double texRatio = (double)Observer.TextureWidth / Observer.TextureHeight;

                switch (Observer.ViewMode)
                {
                    case MyAbstractObserver.ViewMethod.Fit_2D:
                        {
                            double halfWidth = Observer.TextureWidth * MyDefaultShape.SCALE * 0.5;
                            double halfHeight = Observer.TextureHeight * MyDefaultShape.SCALE * 0.5;

                            if (Observer.KeepRatio)
                            {                               
                                if (windowRatio > texRatio)
                                {
                                    halfWidth = halfHeight * windowRatio;                                 
                                }
                                else
                                {                            
                                    halfHeight = halfWidth / windowRatio;
                                }
                            }                            

                            GL.Ortho(-halfWidth, halfWidth, -halfHeight, halfHeight, -1.0, 1.0);                            
                            m_camera = null;                         
                        }
                        break;

                    case MyAbstractObserver.ViewMethod.Free_2D:
                        {
                            Matrix4 m_prespective = Matrix4.CreatePerspectiveFieldOfView((float)Math.PI * 0.25f, (float)windowRatio, 0.1f, 400.0f);
                            GL.LoadMatrix(ref m_prespective);

                            if (!(m_camera is MyZoomAndPanCamera))
                            {
                                m_camera = new MyZoomAndPanCamera(glControl);
                            }
                        }
                        break;

                    case MyAbstractObserver.ViewMethod.Orbit_3D:
                        {                            
                            Matrix4 m_prespective = Matrix4.CreatePerspectiveFieldOfView((float)Math.PI * 0.25f, (float)windowRatio, 0.1f, 400.0f);
                            GL.LoadMatrix(ref m_prespective);                            

                            if (!(m_camera is MyOrbitCamera))
                            {
                                m_camera = new MyOrbitCamera(glControl);
                            }
                        }
                        break;
                }

                if (m_camera != null && Observer.CameraData != null 
                    && Observer.CameraData.CameraType == Observer.ViewMode)
                {
                    m_camera.CameraSetup = Observer.CameraData;
                    Observer.CameraData = null;
                }
            }

            GL.Viewport(0, 0, glControl.Width, glControl.Height);
        }

        private float2 UnprojectFit2DView(int screenX, int screenY)
        {
            float windowRatio = (float)glControl.Width / glControl.Height;
            float texRatio = (float)Observer.TextureWidth / Observer.TextureHeight;

            float width = Observer.TextureWidth;
            float height = Observer.TextureHeight;
            
            if (Observer.KeepRatio)
            {
                if (windowRatio > texRatio)
                {
                    width = height * windowRatio;
                }
                else
                {
                    height = width / windowRatio;
                }
            }

            return new float2(
                ((float)screenX / glControl.Width - 0.5f) * width + Observer.TextureWidth * 0.5f, 
                ((float)screenY / glControl.Height - 0.5f) * height + Observer.TextureHeight * 0.5f
            );
        }

        void SimulationHandler_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            if (!IsDisposed)
            {
                uint simulationStep = (sender as MySimulationHandler).SimulationStep;
                this.Invoke((MethodInvoker)(() => UpdateView(simulationStep)));
            }
        }

        void SimulationHandler_StateChanged(object sender, MySimulationHandler.StateEventArgs e)
        {
            Active = e.NewState != MySimulationHandler.SimulationState.STOPPED;

            if (!Active)
            {
                peekLabel.Visible = false;
            }

            CloseButton = 
                e.NewState == MySimulationHandler.SimulationState.PAUSED || 
                e.NewState == MySimulationHandler.SimulationState.STOPPED;

            if (Observer != null)
            {
                Observer.Active = Active;                
            }

            Observer.TriggerViewReset();
            glControl.Invalidate();

            updateViewToolStripMenuItem.Enabled = 
            snapshotToolStripMenuItem.Enabled = 
                e.NewState == MySimulationHandler.SimulationState.PAUSED;
        }

        private void glControl_Paint(object sender, PaintEventArgs e)
        {
            if (!m_initialized) return;
            
            glControl.MakeCurrent();

            if (Observer.ViewResetNeeded)
            {
                Observer.ViewResetNeeded = false;
                SetupViewport();
            }
            RenderFrame();
        }

        private void glControl_Resize(object sender, EventArgs e)
        {
            if (!m_initialized) return;
            
            glControl.MakeCurrent();
            SetupViewport();
            glControl.Invalidate();            
        }

        private void ObserverForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            m_mainForm.SimulationHandler.StateChanged -= SimulationHandler_StateChanged;
            m_mainForm.SimulationHandler.ProgressChanged -= SimulationHandler_ProgressChanged;
            Observer.Dispose();
            if (m_mainForm.ConsoleView != null)
                m_mainForm.ConsoleView.Focus();
            else
                m_mainForm.Focus(); // to have some valid focus - prevents "minimization" on close
            m_mainForm.RemoveObserverView(this);
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            return m_mainForm.PerformMainMenuClick(keyData);
        }

        protected void ShowValueAt(int x, int y)
        {
            if (Observer is MyMemoryBlockObserver)
            {                
                MyMemoryBlockObserver mbObserver = (Observer as MyMemoryBlockObserver);                                               
                float2 pixelPos = UnprojectFit2DView(x, y);

                if (pixelPos.x > 0 && pixelPos.x < Observer.TextureWidth && pixelPos.y > 0 && pixelPos.y < Observer.TextureHeight)
                {
                    int px = (int)pixelPos.x;
                    int py = (int)pixelPos.y;
                    int index = py * Observer.TextureWidth + px;

                    if (index >= mbObserver.Target.Count)
                        return;

                    peekLabel.Visible = true;

                    float result = 0;
                    mbObserver.Target.GetValueAt(ref result, index);

                    string formattedValue;
                    if (mbObserver.Method == RenderingMethod.Raw)
                    {
                        IEnumerable<string> channels = BitConverter.GetBytes(result).Reverse()  // Get the byte values.
                            .Select(channel => channel.ToString())
                            .Select(channel => new String(' ', 3 - channel.Length) + channel);  // Indent with spaces.

                        // Zip labels and values, join with a separator.
                        formattedValue = String.Join(", ", "ARGB".Zip(channels, (label, channel) => label + ":" + channel));
                    }
                    else
                    {
                        formattedValue = result.ToString("0.0000");
                    }

                    // Show coordinates or index.
                    string formattedIndex = mbObserver.ShowCoordinates ? px + ", " + py : index.ToString();

                    peekLabel.Text = mbObserver.Target.Name + @"[" + formattedIndex + @"] = " + formattedValue;
                }
            }
        }

        private void snapshotToolStripMenuItem_Click(object sender, EventArgs e)
        {           
            Bitmap snapshot = Observer.GrabObserverTexture();
            String docPath = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments) + "\\bs_snapshots";
            Directory.CreateDirectory(docPath);

            string filename = docPath + "\\snapshot_" + Observer.TargetIdentifier + "_step_" + Observer.SimulationStep + ".png";

            snapshot.Save(filename, System.Drawing.Imaging.ImageFormat.Png);
            snapshot.Dispose();

            MyLog.INFO.WriteLine("Snapshot saved to \"" + filename + "\"");
        }

        private Point m_lastMousePosition;

        private void glControl_MouseUp(object sender, MouseEventArgs e)
        {
            if (m_lastMousePosition == e.Location) {

                if ((e.Button & MouseButtons.Left) > 0 && Observer.ViewMode == MyAbstractObserver.ViewMethod.Fit_2D)
                {
                    ShowValueAt(e.Location.X, e.Location.Y);                                     
                }
                else if ((e.Button & MouseButtons.Right) > 0)
                {
                    contextMenuStrip.Show(glControl, e.Location.X + 2, e.Location.Y + 2);
                }         
            }
        }

        private void glControl_MouseDown(object sender, MouseEventArgs e)
        {
            m_lastMousePosition = e.Location;  
        }

        private void updateViewToolStripMenuItem_Click(object sender, EventArgs e)
        {            
            UpdateView(m_mainForm.SimulationHandler.SimulationStep);
        }

        private void goToNodeToolStripMenuItem_Click(object sender, EventArgs e)
        {
            MyNode targetNode = null;
            if(Observer.GenericTarget is MyNode)
            {
                targetNode = Observer.GenericTarget as MyNode;
            }
            else if (Observer.GenericTarget is MyAbstractMemoryBlock) 
            {
                targetNode = (Observer.GenericTarget as MyAbstractMemoryBlock).Owner;
            }

            if (targetNode != null) 
            {
                if (targetNode is MyWorld)
                {
                    GraphLayoutForm graphForm = m_mainForm.OpenGraphLayout(targetNode.Owner.Network);
                    graphForm.worldButton_Click(sender, EventArgs.Empty);
                }
                else
                {
                    GraphLayoutForm graphForm = m_mainForm.OpenGraphLayout(targetNode.Parent);
                    graphForm.SelectNodeView(targetNode);
                }
            }
        }        
    }
}
