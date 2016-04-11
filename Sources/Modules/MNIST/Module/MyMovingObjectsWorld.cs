using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using MNIST;
using System;
using System.ComponentModel;
using YAXLib;

namespace BrainSimulator.Retina
{
    /// <author>GoodAI</author>
    /// <meta>df</meta>
    ///<status>Working</status>
    ///<summary>MNIST numbers moving accross the screen.
    ///Number of objects, sizes, durations and velocities can be adjusted.</summary>
    ///<description></description>
    public class MyMovingObjectsWorld : MyWorld
    {
        public struct MyWorldObject
        {
            public float2 position;
            public float size;
            public int life;
            public float2 velocity;
            public CUdeviceptr bitmap;
        };

        public MyMemoryBlock<MyWorldObject> WorldObjects { get; private set; }
        public MyMemoryBlock<float> Bitmaps { get; private set; }

        [MyOutputBlock]
        public MyMemoryBlock<float> Visual
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 256)]
        public int VisualWidth { get; set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 256)]
        public int VisualHeight { get; set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 5)]
        public int MaxObjects { get; set; }

        private MyMNISTManager m_MNISTManager;
        private static int IMG_WIDTH = 28;

        public MyUpdateObjectsTask UpdateObjects { get; private set; }
        public MyRenderObjectsTask RenderObjects { get; private set; }

        public MyMovingObjectsWorld()
        {
            m_MNISTManager = new MyMNISTManager(MyResources.GetMyAssemblyPath() + @"\res\", 1000);
            m_MNISTManager.RandomEnumerate = true;
        }

        public override void UpdateMemoryBlocks()
        {
            WorldObjects.Count = MaxObjects;

            Bitmaps.Count = MaxObjects * IMG_WIDTH * IMG_WIDTH;
            Bitmaps.ColumnHint = IMG_WIDTH;

            Visual.Count = VisualWidth * VisualHeight;
            Visual.ColumnHint = VisualWidth;
        }

        public class MyUpdateObjectsTask : MyTask<MyMovingObjectsWorld>
        {
            [MyBrowsable, Category("Probalility")]
            [YAXSerializableField(DefaultValue = 0.01f)]
            public float CreateObjectChance { get; set; }

            [MyBrowsable, Category("Probalility")]
            [YAXSerializableField(DefaultValue = 0.5f)]
            public float StaticObjectChance { get; set; }

            [MyBrowsable, Category("Life")]
            [YAXSerializableField(DefaultValue = 500)]
            public float MinLife { get; set; }

            [MyBrowsable, Category("Life")]
            [YAXSerializableField(DefaultValue = 1000)]
            public int MaxLife { get; set; }

            [MyBrowsable, Category("Velocity")]
            [YAXSerializableField(DefaultValue = 0)]
            public float MinVelocity { get; set; }

            [MyBrowsable, Category("Velocity")]
            [YAXSerializableField(DefaultValue = 0.02f)]
            public float MaxVelocity { get; set; }

            [MyBrowsable, Category("Size")]
            [YAXSerializableField(DefaultValue = 0.05f)]
            public float MinSize { get; set; }

            [MyBrowsable, Category("Size")]
            [YAXSerializableField(DefaultValue = 0.1f)]
            public float MaxSize { get; set; }

            private Random m_random = new Random();
            private int[] m_validNumbers = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            public override void Execute()
            {
                bool copyBitmaps = false;

                for (int i = 0; i < Owner.MaxObjects; i++)
                {
                    if (Owner.WorldObjects.Host[i].life > 0)
                    {
                        Owner.WorldObjects.Host[i].position += Owner.WorldObjects.Host[i].velocity;

                        float2 pos = Owner.WorldObjects.Host[i].position;
                        float size = Owner.WorldObjects.Host[i].size;

                        if (pos.x < -1 - size || pos.x > 1 + size || pos.y < -1 - size || pos.y > 1 + size)
                        {
                            Owner.WorldObjects.Host[i].life = 0;
                        }
                        else
                        {
                            Owner.WorldObjects.Host[i].life--;
                        }
                    }

                    if (Owner.WorldObjects.Host[i].life <= 0 && m_random.NextDouble() < CreateObjectChance)
                    {
                        MyWorldObject newObject = new MyWorldObject();

                        newObject.life = (int)(m_random.NextDouble() * MaxLife);
                        newObject.size = (float)m_random.NextDouble() * (MaxSize - MinSize) + MinSize;
                        newObject.position = new float2((float)m_random.NextDouble() * 2 - 1, (float)m_random.NextDouble() * 2 - 1);

                        if (m_random.NextDouble() > StaticObjectChance)
                        {
                            newObject.velocity = new float2((float)m_random.NextDouble() * 2 - 1, (float)m_random.NextDouble() * 2 - 1);

                            float velocitySize = (float)(
                                (m_random.NextDouble() * (MaxVelocity - MinVelocity) + MinVelocity) /
                                Math.Sqrt(newObject.velocity.x * newObject.velocity.x + newObject.velocity.y * newObject.velocity.y)
                            );
                            newObject.velocity *= velocitySize;
                        }

                        MyMNISTImage image = Owner.m_MNISTManager.GetNextImage(m_validNumbers, MNIST.MNISTSetType.Training);
                        Array.Copy(image.Data1D, 0, Owner.Bitmaps.Host, i * IMG_WIDTH * IMG_WIDTH, IMG_WIDTH * IMG_WIDTH);
                        copyBitmaps = true;
                        CudaDeviceVariable<float> devBitmaps = Owner.Bitmaps.GetDevice(Owner);

                        newObject.bitmap = devBitmaps.DevicePointer + (devBitmaps.TypeSize * i * IMG_WIDTH * IMG_WIDTH);

                        Owner.WorldObjects.Host[i] = newObject;
                    }
                }

                Owner.WorldObjects.SafeCopyToDevice();

                if (copyBitmaps)
                {
                    Owner.Bitmaps.SafeCopyToDevice();
                }
            }

            public override void Init(int nGPU)
            {
            }
        };

        public class MyRenderObjectsTask : MyTask<MyMovingObjectsWorld>
        {
            private MyCudaKernel m_kernel;

            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\Transform2DKernels", "BilinearAddSubImageKernel");
                m_kernel.SetupExecution(Owner.Visual.Count);
            }

            public override void Execute()
            {
                Owner.Visual.Fill(0);

                for (int i = 0; i < Owner.MaxObjects; i++)
                {
                    MyWorldObject wo = Owner.WorldObjects.Host[i];

                    if (wo.life > 0)
                    {
                        CudaDeviceVariable<MyWorldObject> devWO = Owner.WorldObjects.GetDevice(Owner);
                        m_kernel.Run(Owner.Visual, wo.bitmap, devWO.DevicePointer + devWO.TypeSize * i, Owner.VisualWidth, Owner.VisualHeight, IMG_WIDTH, IMG_WIDTH);
                    }
                }
            }
        };
    }
}