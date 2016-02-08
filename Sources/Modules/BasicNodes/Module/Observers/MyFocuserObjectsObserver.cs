using GoodAI.Core;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using GoodAI.Modules.Retina;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using System;
using System.ComponentModel;
using System.Drawing;
using GoodAI.Core.Memory;
using YAXLib;

namespace GoodAI.Modules.Observers
{
    public class MyFocuserObjectsObserver : MyNodeObserver<MyFocuser>
    {
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 5)]
        public int FeaturesCount { get; set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 1.0f)]
        public float VelocityScale { get; set; }

        private static readonly Color[] COLOR_TABLE = new Color[] { Color.Red, Color.Green, Color.Navy, Color.Magenta, Color.Cyan, Color.Orange, Color.Lime, Color.Gray };

        public MyFocuserObjectsObserver()
        {
            m_kernel = MyKernelFactory.Instance.Kernel(@"Observers\ColorScaleObserverSingle");

            VelocityScale = 1.0f;
            FeaturesCount = 5;
        }        

        protected override void Execute()
        {
            m_kernel.SetupExecution(Target.InputSize);
            m_kernel.Run(Target.Input, (int)RenderingMethod.GrayScale, (int)MyMemoryBlockObserver.RenderingScale.Linear,
                0, 1.0f, VBODevicePointer, Target.Input.Count);
        }        

        protected override void Reset()
        {
            TextureWidth = Target.Input.ColumnHint;
            TextureHeight = Target.InputSize / TextureWidth;

            Shapes.Add(new MyDefaultShape());
            Shapes.Add(new MyObjectsShape(this));
        }

        private class MyObjectsShape : MyShape
        {
            private MyFocuserObjectsObserver m_owner;

            public MyObjectsShape(MyFocuserObjectsObserver owner)
            {
                m_owner = owner;
            }

            public override void Render()
            {
                if (m_owner.Target.PupilControl.Count % m_owner.FeaturesCount == 0)
                {                    
                    m_owner.Target.PupilControl.SafeCopyToHost();
                    float[] objectsDef = m_owner.Target.PupilControl.Host;
                    int numOfObjects = objectsDef.Length / m_owner.FeaturesCount;

                    GL.PushAttrib(AttribMask.EnableBit);

                    GL.Disable(EnableCap.Texture2D);
                    GL.Disable(EnableCap.Lighting);

                    double sx = m_owner.TextureWidth * MyDefaultShape.SCALE * 0.5;
                    double sy = m_owner.TextureHeight * MyDefaultShape.SCALE * 0.5;

                    GL.Scale(sx, -sy, 1.0f);

                    for (int i = 0; i < numOfObjects; i++)
                    {
                        GL.Color3(COLOR_TABLE[i % COLOR_TABLE.Length]);

                        if (m_owner.FeaturesCount >= 2)
                        {
                            Vector3 position = new Vector3(objectsDef[i * m_owner.FeaturesCount], objectsDef[i * m_owner.FeaturesCount + 1], 0.1f);

                            GL.PushMatrix();
                            GL.Translate(position);

                            GL.PushMatrix();
                            GL.Scale(0.05f, 0.05f, 0.05f);
                            FillUnitSquare();
                            GL.PopMatrix();

                            //TODO: insert size render here                            
                            if (m_owner.FeaturesCount >= 5)
                            {
                                Vector3 velocity = new Vector3(objectsDef[i * m_owner.FeaturesCount + 3], objectsDef[i * m_owner.FeaturesCount + 4], 0.1f);
                                float length = velocity.Length * m_owner.VelocityScale;

                                GL.PushMatrix();

                                GL.Rotate(Math.Atan2(velocity.Y, velocity.X) * 180.0 / Math.PI, 0, 0, 1);                                                                
                                DrawVectorX(length);

                                GL.PopMatrix();
                            }
                            

                            GL.PopMatrix();
                        }                       
                    }

                    GL.PopAttrib();
                }
            }

            private void DrawUnitSquare()
            {
                GL.Begin(PrimitiveType.LineLoop);

                GL.Vertex2(-0.5f, -0.5f);
                GL.Vertex2(-0.5f, 0.5f);
                GL.Vertex2(0.5f, 0.5f);
                GL.Vertex2(0.5f, -0.5f);

                GL.End();
            }

            private void FillUnitSquare()
            {
                GL.Begin(PrimitiveType.Quads);

                GL.Vertex2(-0.5f, -0.5f);
                GL.Vertex2(-0.5f, 0.5f);
                GL.Vertex2(0.5f, 0.5f);
                GL.Vertex2(0.5f, -0.5f);

                GL.End();
            }

            private void DrawVectorX(float length)
            {
                GL.Begin(PrimitiveType.Lines);

                GL.Vertex2(0, 0);
                GL.Vertex2(length, 0);                

                GL.End();

                GL.Begin(PrimitiveType.Triangles);

                GL.Vertex2(length - 0.1f, 0.03f);
                GL.Vertex2(length, 0f);
                GL.Vertex2(length - 0.1f, -0.03f);

                GL.End();
            }
        }
    }
}
