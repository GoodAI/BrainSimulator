using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Newtonsoft.Json;


namespace  XmlFeedForwardNet.Utils.WeightLoader
{
    public class MyConvNetJSWeightLoader
    {
        public List<float[]> ConvWeight;
        public List<float[]> ConvBias;
        public List<float[]> NeuronWeight;
        public List<float[]> NeuronBias;
        public List<float[]> LinearWeight;
        public List<float[]> LinearBias;

        public MyConvNetJSWeightLoader(string path)
        {
            ConvWeight = new List<float[]>();
            ConvBias = new List<float[]>();
            NeuronWeight = new List<float[]>();
            NeuronBias = new List<float[]>();

            parse(path);
        }


        private void parse(string path)
        {
            StreamReader streamReader = new StreamReader(path);
            string jsonContent = streamReader.ReadToEnd();
            streamReader.Close();

            MyConvNetJSWeight net = JsonConvert.DeserializeObject<MyConvNetJSWeight>(jsonContent);

            for (int layerId = 0; layerId < net.layers.Length; layerId++)
            {
                MyConvNetJSWeightLayer layer = net.layers[layerId];
                if (layer.layer_type == "conv") // Convolution layer
                {
                    int kernelWidth = layer.sx;
                    int kernelHeight = layer.sy;
                    int kernelDepth = layer.in_depth;
                    int kernelSize = kernelWidth * kernelHeight * kernelDepth;
                    int kernelNb = layer.out_depth;
                    int weightTotalSize = (kernelSize + 1) * kernelNb;
                    int filterNb = layer.filters.Length;

                    float[] weights = new float[weightTotalSize];
                    float[] biases = new float[kernelNb];

                    // Neuronal weights
                    for (int filterId = 0; filterId < filterNb; filterId++)
                    {
                        MyConvNetJSWeightLayerVol vol = layer.filters[filterId];
                        for (int y = 0; y < kernelHeight; y++)
                            for (int x = 0; x < kernelWidth; x++)
                                for (int z = 0; z < kernelDepth; z++)
                                {
                                    int addr = y * (kernelWidth * kernelDepth) + x * (kernelDepth) + z;
                                    weights[filterId * kernelSize + z * (kernelWidth * kernelHeight) + y * (kernelWidth) + x] = (float)vol.w[addr.ToString()];
                                }
                    }

                    // Biases
                    for (int filterId = 0; filterId < filterNb; filterId++)
                    {
                        biases[filterId] = (float)layer.biases.w[filterId.ToString()];
                    }

                    ConvWeight.Add(weights);
                    ConvBias.Add(biases);
                }
                else if (layer.layer_type == "fc") // Fully connected layer
                {
                    int nbNeurons = layer.out_depth;
                    int nbInputPerNeuron = layer.num_inputs;
                    int weightTotalSize = nbNeurons * (nbInputPerNeuron + 1);

                    float[] weights = new float[weightTotalSize];
                    float[] biases = new float[nbNeurons];

                    MyConvNetJSWeightLayer previousLayer = net.layers[layerId - 1];

                    // Neuronal weights
                    for (int neuronId = 0; neuronId < layer.filters.Length; neuronId++)
                    {
                        MyConvNetJSWeightLayerVol vol = layer.filters[neuronId];
                        for (int y = 0; y < previousLayer.out_sy; y++)
                            for (int x = 0; x < previousLayer.out_sx; x++)
                                for (int z = 0; z < previousLayer.out_depth; z++)
                                {
                                    int addr = y * (previousLayer.out_sx * previousLayer.out_depth) + x * (previousLayer.out_depth) + z;
                                    weights[neuronId * nbInputPerNeuron + z * (previousLayer.out_sy * previousLayer.out_sx) + y * (previousLayer.out_sx) + x] = (float)vol.w[addr.ToString()];
                                }
                    }

                    // Biases
                    for (int filterId = 0; filterId < layer.filters.Length; filterId++)
                    {
                        biases[filterId] = (float)layer.biases.w[filterId.ToString()];
                    }

                    NeuronWeight.Add(weights);
                    NeuronBias.Add(biases);
                }
                else
                {
                    // Nothing. Not interested.
                }
            }
        }
    }
}
