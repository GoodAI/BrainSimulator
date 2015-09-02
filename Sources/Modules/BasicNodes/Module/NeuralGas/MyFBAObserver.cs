using GoodAI.Core;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using ManagedCuda;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using System;
using System.ComponentModel;
using System.Drawing;
using YAXLib;

namespace GoodAI.Modules.NeuralGas
{
    public class MyFBAObserver : MyNodeObserver<MyGrowingNeuralGasNode>
    {

        public enum Option
        { 
            True,
            False
        }

        [MyBrowsable, Category("Init")]
        [YAXSerializableField(DefaultValue = 0.00f), YAXElementFor("Structure")]
        public float COORDINATES_MIN { get; set; }

        [MyBrowsable, Category("Init")]
        [YAXSerializableField(DefaultValue = 1.00f), YAXElementFor("Structure")]
        public float COORDINATES_MAX { get; set; }



        [MyBrowsable, Category("Parameters")]
        [YAXSerializableField(DefaultValue = 0.10f), YAXElementFor("Structure")]
        public float REPULSION { get; set; }

        [MyBrowsable, Category("Parameters")]
        [YAXSerializableField(DefaultValue = 5.00f), YAXElementFor("Structure")]
        public float REPULSION_DISTANCE { get; set; }

        [MyBrowsable, Category("Parameters")]
        [YAXSerializableField(DefaultValue = 0.50f), YAXElementFor("Structure")]
        public float SPRING_STRENGTH { get; set; }

        [MyBrowsable, Category("Parameters")]
        [YAXSerializableField(DefaultValue = 0.10f), YAXElementFor("Structure")]
        public float FORCE_FACTOR { get; set; }

        [MyBrowsable, Category("Parameters")]
        [YAXSerializableField(DefaultValue = Option.False), YAXElementFor("Structure")]
        public Option ONE_SHOT_RESTART { get; set; }


        [MyBrowsable, Category("Parameters")]
        [YAXSerializableField(DefaultValue = Option.False), YAXElementFor("Structure")]
        public Option TRANSLATE_TO_CENTER { get; set; }

        [MyBrowsable, Category("Parameters")]
        [YAXSerializableField(DefaultValue = 0.10f), YAXElementFor("Structure")]
        public float TEXTURE_SIDE { get; set; }

        [MyBrowsable, Category("Parameters")]
        [YAXSerializableField(DefaultValue = 0.10f), YAXElementFor("Structure")]
        public float TIME_STEP { get; set; }




        [MyBrowsable, Category("Non-valid points")]
        [YAXSerializableField(DefaultValue = 0.00f), YAXElementFor("Structure")]
        public float X_NON_VALID { get; set; }

        [MyBrowsable, Category("Non-valid points")]
        [YAXSerializableField(DefaultValue = 0.00f), YAXElementFor("Structure")]
        public float Y_NON_VALID { get; set; }

        [MyBrowsable, Category("Non-valid points")]
        [YAXSerializableField(DefaultValue = 0.00f), YAXElementFor("Structure")]
        public float Z_NON_VALID { get; set; }

        
        private CudaDeviceVariable<float> m_d_PointsCoordinates;
        private CudaDeviceVariable<float> m_d_Force;
        private CudaDeviceVariable<float> m_d_Velocity;
        private CudaDeviceVariable<float> m_d_CubeOperation;
        private CudaDeviceVariable<float> m_d_CubeTexCoordinates;

        private CudaDeviceVariable<float> m_d_CenterOfGravity;

        private CudaDeviceVariable<int> m_d_ActiveConnectionsCount;
        private int[] m_h_ActiveConnectionsCount;


        bool initializedFlag;
        Random randomNumber;
        float actualVelocity;

        private int maxConnections;

        // physics kernels

        private MyCudaKernel m_setForcesToZeroKernel;
        private MyCudaKernel m_computeSpringsKernel;
        private MyCudaKernel m_computeElectricRepulsionKernel;
        private MyCudaKernel m_useForceKernel;

        private MyCudaKernel m_centerOfGravityKernel;

        private MyCudaKernel m_cSpringsKernel;
        private MyCudaKernel m_cRepulsiveKernel;

        private MyCudaKernel m_springKernel;
        private MyCudaKernel m_repulsionKernel;

        // graphics kernels
        private MyCudaKernel m_copyPointsCoordinatesKernel;
        private MyCudaKernel m_copyConnectionsCoordinatesKernel;
        private MyCudaKernel m_computeQuadsKernel;
        private MyCudaKernel m_computeCubesKernel;
        private MyCudaKernel m_computeCubes2Kernel;
        private MyCudaKernel m_cubeCoordinatesKernel;
        private MyCudaKernel m_cubeTextureKernel;
        private MyCudaKernel m_copyAndProcessTextureKernel;
        private MyCudaKernel m_winnersKernel;
        private MyCudaKernel m_zeroTextureKernel;

        private MyBufferedPrimitive m_Connections;
        private MyBufferedPrimitive m_ReferenceFields;
        private MyBufferedPrimitive m_Winners;
        private MyBufferedPrimitive m_WinnerOne;
        private MyBufferedPrimitive m_WinnerTwo;
        private Vector3 m_Translation;

        public MyFBAObserver()
        {
            randomNumber = new Random();
            initializedFlag = false;


            REPULSION = 0.10f;
            REPULSION_DISTANCE = 5.00f;
            SPRING_STRENGTH = 0.50f;
            FORCE_FACTOR = 0.10f;
            TIME_STEP = 0.10f;
            TRANSLATE_TO_CENTER = Option.False;
            TEXTURE_SIDE = 0.10f;

            COORDINATES_MIN = 0.00f;
            COORDINATES_MAX = 1.00f;
            TEXTURE_SIDE = 0.10f;

            X_NON_VALID = 0.00f;
            Y_NON_VALID = 0.00f;
            Z_NON_VALID = 0.00f;

            

            // phyisics kernels
            m_setForcesToZeroKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"GrowingNeuralGas\FBAObserverKernel", "SetForcesToZeroKernel");
            m_springKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"GrowingNeuralGas\FBAObserverKernel", "SpringKernel");
            m_repulsionKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"GrowingNeuralGas\FBAObserverKernel", "RepulsionKernel");
            m_useForceKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"GrowingNeuralGas\FBAObserverKernel", "UseForceKernel");

            m_centerOfGravityKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"GrowingNeuralGas\FBAObserverKernel", "CenterOfGravityKernel");

            // graphics kernels
            m_copyPointsCoordinatesKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"GrowingNeuralGas\FBAObserverKernel", "CopyPointsCoordinatesKernel");
            m_copyConnectionsCoordinatesKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"GrowingNeuralGas\FBAObserverKernel", "CopyConnectionsCoordinatesKernel");
            m_computeQuadsKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"GrowingNeuralGas\FBAObserverKernel", "ComputeQuadsKernel");
            m_computeCubesKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"GrowingNeuralGas\FBAObserverKernel", "ComputeCubesKernel");
            m_computeCubes2Kernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"GrowingNeuralGas\FBAObserverKernel", "ComputeCubes2Kernel");
            m_cubeCoordinatesKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"GrowingNeuralGas\FBAObserverKernel", "CubeCoordinatesKernel");
            m_cubeTextureKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"GrowingNeuralGas\FBAObserverKernel", "CubeTextureKernel");
            m_copyAndProcessTextureKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"GrowingNeuralGas\FBAObserverKernel", "CopyAndProcessTextureKernel");
            m_winnersKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"GrowingNeuralGas\FBAObserverKernel", "WinnersKernel");
            m_zeroTextureKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"GrowingNeuralGas\FBAObserverKernel", "ZeroTextureKernel");
            actualVelocity = 1.00f;

            m_h_ActiveConnectionsCount = new int[1];
        }

        protected override void Execute()
        {
            



            if (!initializedFlag)
            {
                m_d_Force = new CudaDeviceVariable<float>(Target.MAX_CELLS * 3);
                
                m_d_ActiveConnectionsCount = new CudaDeviceVariable<int>(1);
                m_d_CenterOfGravity = new CudaDeviceVariable<float>(3);
                
                //initialize vertices position
                InitCoordinatesAndVelocity();
                initializedFlag = true;
                ViewMode = ViewMethod.Orbit_3D;

                float translationValue = 0.50f * (COORDINATES_MAX - COORDINATES_MIN);
                m_Translation = new Vector3(-translationValue, 0, -translationValue);

                m_zeroTextureKernel.SetupExecution(TextureHeight * TextureWidth);

                m_zeroTextureKernel.Run(
                    VBODevicePointer,
                    TextureHeight * TextureWidth
                    );

            }


            if (TRANSLATE_TO_CENTER == Option.True)
            {
                m_centerOfGravityKernel.SetupExecution(1);

                m_centerOfGravityKernel.Run(
                    m_d_PointsCoordinates.DevicePointer,
                    m_d_CenterOfGravity.DevicePointer,
                    Target.ActivityFlag,
                    Target.MAX_CELLS
                    );

                float[] m_h_centerOfGravity = new float[3];
                m_d_CenterOfGravity.CopyToHost(m_h_centerOfGravity);

                m_Translation = new Vector3(-m_h_centerOfGravity[0], -m_h_centerOfGravity[1], -m_h_centerOfGravity[2]);

                m_Connections.Translation = m_Translation;
                m_ReferenceFields.Translation = m_Translation;
                m_WinnerOne.Translation = m_Translation;
                m_WinnerTwo.Translation = m_Translation;

            }

            // PHYSICS PART
            // set forces to zero
            m_setForcesToZeroKernel.SetupExecution(Target.MAX_CELLS * 3);

            m_setForcesToZeroKernel.Run(
                m_d_Force.DevicePointer,
                Target.MAX_CELLS
                );

            // spring force computation
            m_springKernel.SetupExecution(Target.MAX_CELLS);

            m_springKernel.Run(
                Target.ActivityFlag,
                Target.ConnectionMatrix,
                m_d_PointsCoordinates.DevicePointer,
                SPRING_STRENGTH,
                m_d_Force.DevicePointer,
                Target.MAX_CELLS
                );
            

            // repulsion force computation
            m_repulsionKernel.SetupExecution(Target.MAX_CELLS);

            m_repulsionKernel.Run(
                REPULSION,
                REPULSION_DISTANCE,
                m_d_Force.DevicePointer,
                m_d_PointsCoordinates.DevicePointer,
                Target.ActivityFlag,
                Target.MAX_CELLS
                );
            

            // applying forces to the points
            m_useForceKernel.SetupExecution(Target.MAX_CELLS * 3);

            m_useForceKernel.Run(
                m_d_Force.DevicePointer,
                FORCE_FACTOR,
                m_d_PointsCoordinates.DevicePointer,
                Target.MAX_CELLS
                );
             
 

            // GRAPHICS PART
            // COPY AND PROCESS TEXTURE
            m_copyAndProcessTextureKernel.SetupExecution(Target.ReferenceVector.Count);

            m_copyAndProcessTextureKernel.Run(
                Target.ReferenceVector,
                Target.INPUT_SIZE,
                Target.Input.ColumnHint,
                TextureWidth,
                VBODevicePointer,
                Target.MAX_CELLS,
                Target.ReferenceVector.Count
                );


            
            // CONNECTIONS
            m_d_ActiveConnectionsCount.CopyToDevice(0);

            m_copyConnectionsCoordinatesKernel.SetupExecution(Target.MAX_CELLS * Target.MAX_CELLS);

            m_copyConnectionsCoordinatesKernel.Run(
                Target.ConnectionMatrix,
                m_d_PointsCoordinates.DevicePointer,
                VertexVBODevicePointer,
                m_d_ActiveConnectionsCount.DevicePointer,
                Target.MAX_CELLS
                );

            m_d_ActiveConnectionsCount.CopyToHost(m_h_ActiveConnectionsCount);
            m_Connections.VertexCount = 2 * m_h_ActiveConnectionsCount[0];

            // REFERENCE VECTORS (CUBES)
            /*
            m_computeCubesKernel.m_kernel.SetupExecution(Target.MAX_CELLS
                );


            .Run(
                m_computeCubesKernel,
                m_d_PointsCoordinates.DevicePointer,
                VertexVBODevicePointer,
                m_ReferenceFields.VertexOffset,
                TEXTURE_SIDE,
                Target.ActivityFlag,
                Target.Input.ColumnHint,
                Target.MAX_CELLS
                );
            */

            /*
            m_cubeCoordinatesKernel.m_kernel.SetupExecution(Target.MAX_CELLS * 72
                );

            .Run(
                m_cubeCoordinatesKernel,
                VertexVBODevicePointer,
                m_d_CubeOperation.DevicePointer,
                m_ReferenceFields.VertexOffset,
                Target.ActivityFlag,
                TEXTURE_SIDE,
                m_d_PointsCoordinates.DevicePointer,
                Target.MAX_CELLS
                );

            m_cubeTextureKernel.m_kernel.SetupExecution(Target.MAX_CELLS * 48
                );

            .Run(
                m_cubeTextureKernel,
                VertexVBODevicePointer,
                m_ReferenceFields.TexCoordOffset,
                m_d_CubeTexCoordinates.DevicePointer,
                TEXTURE_SIDE,
                Target.Input.ColumnHint,
                Target.ActivityFlag,
                Target.MAX_CELLS
                );
            */
            
            m_computeCubes2Kernel.SetupExecution(Target.MAX_CELLS * 6);

            m_computeCubes2Kernel.Run(
                m_d_PointsCoordinates.DevicePointer,
                VertexVBODevicePointer,
                m_ReferenceFields.VertexOffset,
                TEXTURE_SIDE,
                m_d_CubeOperation.DevicePointer,
                m_d_CubeTexCoordinates.DevicePointer,
                Target.ActivityFlag,
                (float)Target.Input.ColumnHint,
                Target.MAX_CELLS
                );
            


            /*
            m_computeQuadsKernel.m_kernel.SetupExecution(
                Target.MAX_CELLS
                );

            m_computeQuadsKernel.Run(
                m_d_PointsCoordinates.DevicePointer,
                VertexVBODevicePointer,
                m_ReferenceFields.VertexOffset,
                TEXTURE_SIDE,
                Target.ActivityFlag,
                Target.Input.ColumnHint,
                Target.MAX_CELLS
                );
             */

            m_winnersKernel.SetupExecution(Target.MAX_CELLS);

            m_winnersKernel.Run(
                Target.WinnerOne,
                VertexVBODevicePointer,
                m_WinnerOne.VertexOffset,
                m_d_PointsCoordinates.DevicePointer,
                TEXTURE_SIDE,
                Target.MAX_CELLS
                );

            m_winnersKernel.SetupExecution(Target.MAX_CELLS);

            m_winnersKernel.Run(
                Target.WinnerTwo,
                VertexVBODevicePointer,
                m_WinnerTwo.VertexOffset,
                m_d_PointsCoordinates.DevicePointer,
                TEXTURE_SIDE,
                Target.MAX_CELLS
                );

            if (ONE_SHOT_RESTART == Option.True)
            {
                initializedFlag = false;
                TriggerReset();
                ONE_SHOT_RESTART = Option.False;
            }

        }

        void InitCoordinatesAndVelocity()
        {
            m_d_PointsCoordinates = new CudaDeviceVariable<float>(3 * Target.MAX_CELLS);
            float[] m_h_pointsCoordinates = new float[3 * Target.MAX_CELLS];
            for (int c = 0; c < m_h_pointsCoordinates.Length; c++)
            {
                m_h_pointsCoordinates[c] = COORDINATES_MIN + (COORDINATES_MAX - COORDINATES_MIN) * (float)randomNumber.NextDouble();
                //m_h_pointsCoordinates[c] = 1.00f;
            }
            m_d_PointsCoordinates.CopyToDevice(m_h_pointsCoordinates);

            m_d_Velocity = new CudaDeviceVariable<float>(3 * Target.MAX_CELLS);
            float[] m_h_velocity = new float[3 * Target.MAX_CELLS];
            for (int c = 0; c < m_h_velocity.Length; c++)
            {
                m_h_velocity[c] = 0.00f;
            }
            m_d_Velocity.CopyToDevice(m_h_velocity);




            m_d_CubeOperation = new CudaDeviceVariable<float>(6 * 4 * 3);
            int[] operationMask = new int[6 * 4 * 3] 
                { -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, +1,
                  -1, +1, +1, -1, -1, +1, +1, -1, +1, +1, +1, +1,
                  -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, +1, +1,
                  -1, +1, -1, -1, -1, -1, +1, -1, -1, +1, +1, -1,
                  +1, +1, -1, +1, -1, -1, +1, -1, +1, +1, +1, +1,
                  -1, +1, +1, -1, +1, -1, +1, +1, -1, +1, +1, +1};
            float[] m_h_CubeOperation = new float[6 * 4 * 3];
            for (int i = 0; i < operationMask.Length; i++)
            {
                m_h_CubeOperation[i] = (float)operationMask[i];
            }
            m_d_CubeOperation.CopyToDevice(m_h_CubeOperation);



            m_d_CubeTexCoordinates = new CudaDeviceVariable<float>(6 * 4 * 2);
            int[] texCoordinates = new int[6 * 4 * 2]
                { 0, 0, 0, 1, 1, 1, 1, 0,
                  0, 0, 0, 1, 1, 1, 1, 0,
                  0, 0, 0, 1, 1, 1, 1, 0,
                  1, 0, 1, 1, 0, 1, 0, 0,
                  1, 0, 1, 1, 0, 1, 0, 0,
                  0, 1, 0, 0, 1, 0, 1, 1 };
            float[] m_h_CubeTexCoordinates = new float[6 * 4 * 2];
            for (int i = 0; i < texCoordinates.Length; i++)
            {
                m_h_CubeTexCoordinates[i] = (float)texCoordinates[i];
            }
            m_d_CubeTexCoordinates.CopyToDevice(m_h_CubeTexCoordinates);

            m_computeCubes2Kernel.SetConstantVariable("operationMaskConstant", m_h_CubeOperation);
            m_computeCubes2Kernel.SetConstantVariable("cubeTexCoordinatesConstant",m_h_CubeTexCoordinates);

        }

        protected override void Reset()
        {
            InitCoordinatesAndVelocity();
            maxConnections = Target.MAX_CELLS * Target.MAX_CELLS - Target.MAX_CELLS;

            TextureWidth = Target.Input.ColumnHint * Target.MAX_CELLS;
            TextureHeight = (int)Math.Ceiling((double)Target.INPUT_SIZE / (double)Target.Input.ColumnHint);
            //TextureHeight = Target.INPUT_SIZE / Target.Input.ColumnHint;

            

            VertexDataSize = maxConnections * 2 * 3;
            VertexDataSize += Target.MAX_CELLS * 4 * 6 * 3;
            //VertexDataSize += Target.MAX_CELLS * 4 * 3 * 3;
            // texture coordinates
            VertexDataSize += Target.MAX_CELLS * 4 * 6 * 2;
            //VertexDataSize += Target.MAX_CELLS * 4 * 3 * 2;

            //VertexDataSize += 4 * 2 * 2 * 3;
            VertexDataSize += 4 * 2 * 2 * 3;

            
            m_Connections = new MyBufferedPrimitive(PrimitiveType.Lines, maxConnections * 2, MyVertexAttrib.Position)
            {
                //VertexOffset = Target.MAX_CELLS * 3,
                VertexOffset = 0,
                BaseColor = Color.FromArgb(0, 0, 0),
                Translation = m_Translation
            };

            m_ReferenceFields = new MyBufferedPrimitive(PrimitiveType.Quads, Target.MAX_CELLS * 24, MyVertexAttrib.Position | MyVertexAttrib.TexCoord)
            //m_ReferenceFields = new MyBufferedPrimitive(PrimitiveType.Quads, Target.MAX_CELLS * 12, MyVertexAttrib.Position | MyVertexAttrib.TexCoord)
            {
                VertexOffset = maxConnections * 6,
                TexCoordOffset =  maxConnections * 6 + Target.MAX_CELLS * 4 * 6 * 3,
                //VertexDataSize - Target.MAX_CELLS * 4 * 6 * 2,
                //TexCoordOffset = VertexDataSize - Target.MAX_CELLS * 4 * 3 * 2,
                //BaseColor = Color.FromArgb(0,0,0),
                Translation = m_Translation
            };

            /*
            m_Winners = new MyBufferedPrimitive(PrimitiveType.Quads, 2 * 2, MyVertexAttrib.Position)
            {
                VertexOffset = VertexDataSize - 4 * 2 * 2 * 3,
                BaseColor = Color.FromArgb(255,0,0),
                Translation = m_Translation
            };
            */

            m_WinnerOne = new MyBufferedPrimitive(PrimitiveType.LineLoop, 4, MyVertexAttrib.Position)
            {
                VertexOffset = m_ReferenceFields.TexCoordOffset + Target.MAX_CELLS * 4 * 6 * 2,
                BaseColor = Color.FromArgb(255,0,0),
                Translation = m_Translation
            };

            m_WinnerTwo = new MyBufferedPrimitive(PrimitiveType.LineLoop, 4, MyVertexAttrib.Position)
            {
                VertexOffset = m_WinnerOne.VertexOffset + 4 * 3,
                BaseColor = Color.FromArgb(0,255,0),
                Translation = m_Translation
            };

            Shapes.Add(m_Connections);
            Shapes.Add(m_ReferenceFields);
            //Shapes.Add(m_Winners);
            Shapes.Add(m_WinnerOne);
            Shapes.Add(m_WinnerTwo);

            
        }
    }
}
