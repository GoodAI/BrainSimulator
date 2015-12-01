using BEPUphysics;
using BEPUphysics.BroadPhaseEntries.MobileCollidables;
using BEPUphysics.CollisionShapes;
using BEPUphysics.CollisionShapes.ConvexShapes;
using BEPUphysics.Entities;
using GoodAI.Core.Observers;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using System;
using System.Drawing;

namespace GoodAI.Modules.Motor
{
    public class My3DWorldObserver : MyNodeObserver<My3DWorld>
    {
        Space m_formerSpace;

        public My3DWorldObserver()
        {
        }

        protected override void Reset()
        {
            base.Reset();

            m_formerSpace = Target.Space;

            Shapes.Add(new MyWorldShape(Target.Space));
        }

        protected override void Execute()
        {
            if (m_formerSpace != Target.Space)
            {
                m_formerSpace = Target.Space;

                Shapes.Clear();
                Shapes.Add(new MyWorldShape(Target.Space));
            }
        }
    }

    public class MyWorldShape : MyShape
    {
        private Space m_space;
        private Vector2[] m_circlePoints;

        public MyWorldShape(Space space)
        {
            m_space = space;
            m_circlePoints = new Vector2[16];

            float dAngle = (float)Math.PI * 2 / m_circlePoints.Length;

            for (int i = 0; i < m_circlePoints.Length; i++)
            {
                float a0 = i * dAngle;
                m_circlePoints[i] = new Vector2((float)Math.Cos(i * dAngle), (float)Math.Sin(i * dAngle));
            }
        }

        public override void Render()
        {
            GL.Disable(EnableCap.Texture2D);
            GL.Enable(EnableCap.Lighting);
            GL.Enable(EnableCap.CullFace);
            GL.CullFace(CullFaceMode.Back);

            for (int i = 0; i < m_space.Entities.Count; i++)
            {
                Entity entity = m_space.Entities[i];

                if (!entity.IsDynamic)
                {
                    GL.Color3(Color.LightGreen);
                }
                else if (entity.Tag is Color)
                {
                    GL.Color3((Color)entity.Tag);
                }
                else
                {
                    GL.Color3(Color.White);
                }

                GL.PushMatrix();

                GL.Scale(0.1f, 0.1f, 0.1f);

                Matrix4 wt = entity.WorldTransform;
                GL.MultMatrix(ref wt);

                RenderEntity(entity.CollisionInformation);

                GL.PopMatrix();
            }
        }

        private void RenderEntity(EntityCollidable entity)
        {
            if (entity is CompoundCollidable)
            {
                CompoundCollidable compound = entity as CompoundCollidable;

                for (int i = 0; i < compound.Children.Count; i++)
                {
                    CompoundChild child = compound.Children[i];
                    Matrix4 lt = child.Entry.LocalTransform.Matrix;
                    GL.MultMatrix(ref lt);

                    RenderEntity(child.CollisionInformation);
                }
            }
            else
            {
                EntityShape shape = entity.Shape;

                if (shape is BoxShape)
                {
                    BoxShape box = shape as BoxShape;
                    RenderCube(box.Width, box.Height, box.Length);
                }
                else if (shape is CylinderShape)
                {
                    CylinderShape cylinder = shape as CylinderShape;
                    RenderCylinder(cylinder.Radius, cylinder.Height);
                }
                else if (shape is ConeShape)
                {
                    ConeShape cone = shape as ConeShape;
                    RenderCone(cone.Radius, cone.Height);
                }
                else if (shape is SphereShape)
                {
                    SphereShape sphere = shape as SphereShape;
                    RenderSphere(sphere.Radius);
                }
            }
        }

        private void RenderCube(float width, float height, float depth)
        {
            float w2 = width * 0.5f;
            float h2 = height * 0.5f;
            float d2 = depth * 0.5f;

            GL.Begin(PrimitiveType.Quads);

            GL.Normal3(0, 0, -1.0f);
            GL.Vertex3(-w2, -h2, -d2);
            GL.Vertex3(-w2, h2, -d2);
            GL.Vertex3(w2, h2, -d2);
            GL.Vertex3(w2, -h2, -d2);

            GL.Normal3(0, 0, 1.0f);
            GL.Vertex3(-w2, -h2, d2);
            GL.Vertex3(w2, -h2, d2);
            GL.Vertex3(w2, h2, d2);
            GL.Vertex3(-w2, h2, d2);

            GL.Normal3(-1.0f, 0, 0);
            GL.Vertex3(-w2, -h2, d2);
            GL.Vertex3(-w2, h2, d2);
            GL.Vertex3(-w2, h2, -d2);
            GL.Vertex3(-w2, -h2, -d2);

            GL.Normal3(1.0f, 0, 0);
            GL.Vertex3(w2, -h2, d2);
            GL.Vertex3(w2, -h2, -d2);
            GL.Vertex3(w2, h2, -d2);
            GL.Vertex3(w2, h2, d2);

            GL.Normal3(0, -1.0f, 0);
            GL.Vertex3(-w2, -h2, -d2);
            GL.Vertex3(w2, -h2, -d2);
            GL.Vertex3(w2, -h2, d2);
            GL.Vertex3(-w2, -h2, d2);

            GL.Normal3(0, 1.0f, 0);
            GL.Vertex3(-w2, h2, -d2);
            GL.Vertex3(-w2, h2, d2);
            GL.Vertex3(w2, h2, d2);
            GL.Vertex3(w2, h2, -d2);

            GL.End();
        }

        private void RenderCylinder(float radius, float height)
        {
            float h2 = height * 0.5f;
            int steps = m_circlePoints.Length;

            GL.Begin(PrimitiveType.QuadStrip);

            for (int i = 0; i <= steps; i++)
            {
                float x0 = radius * m_circlePoints[i % steps].X;
                float z0 = radius * m_circlePoints[i % steps].Y;

                GL.Normal3(x0, 0, z0);
                GL.Vertex3(x0, -h2, z0);
                GL.Vertex3(x0, h2, z0);
            }

            GL.End();

            GL.Begin(PrimitiveType.TriangleFan);

            GL.Normal3(0, 1.0f, 0);
            for (int i = steps - 1; i >= 0; i--)
            {
                float x0 = radius * m_circlePoints[i].X;
                float z0 = radius * m_circlePoints[i].Y;

                GL.Vertex3(x0, h2, z0);
            }

            GL.End();

            GL.Begin(PrimitiveType.TriangleFan);

            GL.Normal3(0, -1.0f, 0);
            for (int i = 0; i < steps; i++)
            {
                float x0 = radius * m_circlePoints[i].X;
                float z0 = radius * m_circlePoints[i].Y;

                GL.Vertex3(x0, -h2, z0);
            }

            GL.End();
        }

        private void RenderCone(float radius, float height)
        {
            float h2 = height * 0.25f;
            int steps = m_circlePoints.Length;

            GL.Begin(PrimitiveType.TriangleFan);

            GL.Normal3(0, -1.0f, 0);
            for (int i = 0; i < steps; i++)
            {
                float x0 = radius * m_circlePoints[i].X;
                float z0 = radius * m_circlePoints[i].Y;

                GL.Vertex3(x0, -h2, z0);
            }

            GL.End();

            float ny = radius / height;

            GL.Begin(PrimitiveType.TriangleFan);

            GL.Normal3(0, 1.0f, 0);
            GL.Vertex3(0, height * 0.75f, 0);

            for (int i = steps; i >= 0; i--)
            {
                float x0 = radius * m_circlePoints[i % steps].X;
                float z0 = radius * m_circlePoints[i % steps].Y;

                GL.Normal3(x0, ny, z0);
                GL.Vertex3(x0, -h2, z0);
            }

            GL.End();
        }

        private void RenderSphere(float radius)
        {
            int steps = m_circlePoints.Length;

            for (int j = 1; j < steps / 2 - 1; j++)
            {
                float h1 = radius * m_circlePoints[j + 1].X;
                float h2 = radius * m_circlePoints[j].X;

                GL.Begin(PrimitiveType.QuadStrip);

                for (int i = 0; i <= steps; i++)
                {
                    float x = radius * m_circlePoints[i % steps].X * m_circlePoints[j + 1].Y;
                    float z = radius * m_circlePoints[i % steps].Y * m_circlePoints[j + 1].Y;

                    GL.Normal3(x, h1, z);
                    GL.Vertex3(x, h1, z);

                    x = radius * m_circlePoints[i % steps].X * m_circlePoints[j].Y;
                    z = radius * m_circlePoints[i % steps].Y * m_circlePoints[j].Y;

                    GL.Normal3(x, h2, z);
                    GL.Vertex3(x, h2, z);
                }

                GL.End();
            }

            GL.Begin(PrimitiveType.TriangleFan);

            GL.Normal3(0, 1.0f, 0);
            GL.Vertex3(0, radius, 0);

            for (int i = steps; i >= 0; i--)
            {
                float y = radius * m_circlePoints[1].X;

                float x = radius * m_circlePoints[i % steps].X * m_circlePoints[1].Y;
                float z = radius * m_circlePoints[i % steps].Y * m_circlePoints[1].Y;

                GL.Normal3(x, y, z);
                GL.Vertex3(x, y, z);
            }

            GL.End();

            GL.Begin(PrimitiveType.TriangleFan);

            GL.Normal3(0, -1.0f, 0);
            GL.Vertex3(0, -radius, 0);

            for (int i = 0; i <= steps; i++)
            {
                float y = -radius * m_circlePoints[1].X;

                float x = radius * m_circlePoints[i % steps].X * m_circlePoints[1].Y;
                float z = radius * m_circlePoints[i % steps].Y * m_circlePoints[1].Y;

                GL.Normal3(x, y, z);
                GL.Vertex3(x, y, z);
            }

            GL.End();
        }
    }
}