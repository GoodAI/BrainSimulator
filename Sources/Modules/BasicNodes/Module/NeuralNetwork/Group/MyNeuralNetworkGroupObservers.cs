using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Observers
{
    /// <author>GoodAI</author>
    /// <meta>hk</meta>
    /// <status>WIP</status>
    /// <summary>MyNeuralNetworkGroupObserver</summary>
    /// <description>
    /// TO DO:
    ///  <ul>
    ///    <li> Add support for parallel architectures. </li>
    ///    <li> Add input layer. </li>
    ///    <li> It needs update to work with buffers!! I'v some expe with OpenGL but cannot do it in OpenTK, so far... </li>
    ///    <li> needs support for diff networks (LSTM, recur...) and etc.. works just with basic... </li>
    ///  </ul>
    /// </description>
    public class MyNeuralNetworkGroupObserver : MyNodeObserver<MyNeuralNetworkGroup>
    {

        public enum MyPointVisMode
        {
            Output,
            Delta
        }
        public enum MyEdgeVisMode
        {
            None,
            Ones,
            Output,
            Weights,
            WeightsXOut
        }

        [MyBrowsable, Category("Operation"), YAXSerializableField(DefaultValue = MyPointVisMode.Output), Description("What neurons shows")]
        public MyPointVisMode PointVisMode { get; set; }
        [MyBrowsable, Category("Operation"), YAXSerializableField(DefaultValue = MyEdgeVisMode.Output), Description("What lines shows")]
        public MyEdgeVisMode EdgeVisMode { get; set; }
        [MyBrowsable, Category("Operation"), YAXSerializableField(DefaultValue = 1.0f), Description("Manual line visibility control.")]
        public float EdgeVisMultiplier { get; set; }

        public MyNeuralNetworkGroupObserver()
        {
        }

        protected override void Reset()
        {
            base.Reset();

            Shapes.Add(new MyArrayOfPointsForNNGroupHelperShape(Target.FirstTopologicalLayer,this));
        }

        protected override void Execute()
        {

        }
    }



    public class MyArrayOfPointsForNNGroupHelperShape : MyShape
    {
        float[] m_data; // postions of neurons (x,y,z for each)
        float[] m_data_input; // postions of neurons (x,y,z for each)
        int m_dataDim = 3; /// jsut x,y,z :)
        Random rnd = new Random();
        MyAbstractLayer m_firstLayer; // poniter to first layer in the NNGRoup
        MyNeuralNetworkGroupObserver ThisObserverObject; // pointer to MyNeuralNetworkGroupObserver

        public MyArrayOfPointsForNNGroupHelperShape(MyAbstractLayer firstLayer, MyNeuralNetworkGroupObserver O)//float[] data, int dataDim, float[] labels)
        {
            ThisObserverObject = O;
            m_firstLayer = firstLayer;
            //--- size of net
            int m_dataLen = 0;
            MyAbstractLayer tmpLayer = firstLayer;
            int Nlayers = 0;
            while (tmpLayer != null)
            {
                m_dataLen += tmpLayer.Neurons;
                tmpLayer = tmpLayer.NextTopologicalLayer;
                ++Nlayers;
            }
            m_data = new float[m_dataLen*3];
            
            //--- init neuron positions
            int layerSpacePosition = -Nlayers/2;
            int neuId = 0;
            tmpLayer = firstLayer;
            while (tmpLayer != null)
            {
                Vector3 layerCenter;
                layerCenter.X = 0.0f;//(float)rnd.Next(1,100); // needed for paralel connections :)
                layerCenter.Y = 0.0f;//(float)rnd.Next(1,100); 
                layerCenter.Z = (float)layerSpacePosition;
                if (tmpLayer.NextTopologicalLayer != null) // normal layer has random postions
                {
                    for (int i = 0; i < tmpLayer.Neurons; i++)
                    {
                        m_data[neuId++] = layerCenter.X + ((float)rnd.Next(0, 100)) / 100.0f;
                        m_data[neuId++] = layerCenter.Y + ((float)rnd.Next(0, 100)) / 100.0f;
                        m_data[neuId++] = layerCenter.Z + ((float)rnd.Next(0, 100)) / 500.0f;
                    }
                }
                else // output is grid
                {
                    int width = (int)Math.Sqrt(tmpLayer.Neurons);
                    bool run = true;
                    for (int i = 0; run; i++)
                    {
                        for (int j = 0; j < width; j++)
                        {
                            if ((i * width + j) >= tmpLayer.Neurons)
                            {
                                run = false;
                                break;
                            }
                            m_data[neuId++] = layerCenter.X + ((float)j)/((float)width);
                            m_data[neuId++] = layerCenter.Y + ((float)i)/((float)width);
                            m_data[neuId++] = layerCenter.Z;
                        }
                        if (!run)
                        {
                            break;
                        }
                    }
                }
                layerSpacePosition++;
                tmpLayer = tmpLayer.NextTopologicalLayer;
            }
        }

        public override void Render()
        {
            GL.ClearColor(.15f, .15f, .15f, 0.0f);
            GL.Enable(EnableCap.AlphaTest);
            GL.Disable(EnableCap.DepthTest);
            //Gl.AlphaFunc(GL_NOTEQUAL, 0);
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.One);
            GL.Enable(EnableCap.PointSmooth);
            
            //////////////////////////////////////////////////////
            //   P O I N T S
            GL.Enable(EnableCap.PointSmooth);
            GL.PointSize(4.0f);
            GL.Begin(PrimitiveType.Points);
            MyAbstractLayer currentLayer = m_firstLayer;
            MyMemoryBlock<float> memBlockData2Vis = currentLayer.Output;
            bool normalizeData2Vis = false;
            int a = 0;
            while (currentLayer != null)
            {
                //--- which data to dispaly
                switch (ThisObserverObject.PointVisMode)
                {
                    case MyNeuralNetworkGroupObserver.MyPointVisMode.Output:
                        memBlockData2Vis = currentLayer.Output;
                        break;
                    case MyNeuralNetworkGroupObserver.MyPointVisMode.Delta:
                        memBlockData2Vis = currentLayer.Delta;
                        normalizeData2Vis = true;
                        break;
                    default:
                        memBlockData2Vis = currentLayer.Output;
                        break;
                }
                memBlockData2Vis.SafeCopyToHost();
                //--- go through neurons and plot each
                for (int j = 0; j < currentLayer.Neurons; j++) // this is super in efficint :(
                {
                    int id = a * m_dataDim;
                    float val = memBlockData2Vis.Host[j];
                    val = Math.Abs(val); // value has to be >0
                    if (normalizeData2Vis) // norlaimnze deltas :)
                    {
                        val = val*3f;
                    }
                    System.Drawing.Color col = MyObserverHelpers.ColorFromHSV(120f, 0.5f, Math.Min(val,1.0f)); // value has to be 0-1
                    GL.Color3(col.R / 256f, col.G / 256f, col.B / 256f);
                    GL.Vertex3(m_data[id], m_data[id + 1], m_data[id + 2]);
                    ++a;
                }
                MyAbstractLayer nextLayer = currentLayer.NextTopologicalLayer;
                currentLayer = currentLayer.NextTopologicalLayer;
            }
            GL.End();

            //////////////////////////////////////////////////////
            //    L I N E S
            if (ThisObserverObject.EdgeVisMode != MyNeuralNetworkGroupObserver.MyEdgeVisMode.None)
            {
                GL.LineWidth(0.1f);
                GL.Color4(.2f, .2f, .5f, 0.06f);
                GL.Begin(PrimitiveType.Lines);
                currentLayer = m_firstLayer;
                int curIdxStart = 0; // start of the current index
                while (currentLayer != null)
                {
                    int nextIdxStart = curIdxStart + currentLayer.Neurons;
                    MyAbstractLayer nextLayer = currentLayer.NextTopologicalLayer;
                    if (nextLayer != null)
                    {
                        //--- what to show
                        switch (ThisObserverObject.EdgeVisMode)
                        {
                            case MyNeuralNetworkGroupObserver.MyEdgeVisMode.Ones:
                                break;
                            case MyNeuralNetworkGroupObserver.MyEdgeVisMode.Output:
                                currentLayer.Output.SafeCopyToHost();
                                break;
                            case MyNeuralNetworkGroupObserver.MyEdgeVisMode.Weights:
                                (nextLayer as MyHiddenLayer).Weights.SafeCopyToHost();
                                break;
                            case MyNeuralNetworkGroupObserver.MyEdgeVisMode.WeightsXOut:
                                (nextLayer as MyHiddenLayer).Weights.SafeCopyToHost();
                                currentLayer.Output.SafeCopyToHost();
                                break;
                            default:
                                break;
                        }
                        
                        for (int nc = 0; nc < currentLayer.Neurons; nc++)
                        {
                            for (int nn = 0; nn < nextLayer.Neurons; nn++)
                            {
                                float edgeWeight = .007f;
                                switch (ThisObserverObject.EdgeVisMode)
                                {
                                    case MyNeuralNetworkGroupObserver.MyEdgeVisMode.Output:
                                        edgeWeight = currentLayer.Output.Host[nc] / 50f;
                                        break;
                                    case MyNeuralNetworkGroupObserver.MyEdgeVisMode.Weights:
                                        edgeWeight = (nextLayer as MyHiddenLayer).Weights.Host[nn * currentLayer.Neurons + nc];
                                        break;
                                    case MyNeuralNetworkGroupObserver.MyEdgeVisMode.WeightsXOut:
                                        edgeWeight = currentLayer.Output.Host[nc];
                                        edgeWeight *= (nextLayer as MyHiddenLayer).Weights.Host[nn * currentLayer.Neurons + nc];
                                        break;
                                    default:
                                        break;
                                }
                                int i_c = (nc + curIdxStart) * m_dataDim; // index current
                                int i_n = (nn + nextIdxStart) * m_dataDim; // index next
                                GL.Color4(.5f, .5f, .5f, edgeWeight * ThisObserverObject.EdgeVisMultiplier);
                                GL.Vertex3(m_data[i_c], m_data[i_c + 1], m_data[i_c + 2]);
                                GL.Vertex3(m_data[i_n], m_data[i_n + 1], m_data[i_n + 2]);
                            }
                        }
                    }
                    currentLayer = nextLayer;
                    curIdxStart = nextIdxStart;
                }
                GL.End();
            }
            

        }




    }



}
