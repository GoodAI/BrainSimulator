using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace  XmlFeedForwardNet.Utils.WeightLoader
{
    public class MyConvNetJSWeight
    {
        public MyConvNetJSWeightLayer[] layers;
    }

    public class MyConvNetJSWeightLayer
    {
        public int in_depth;
        public int out_depth;
        public int out_sx;
        public int out_sy;
        public int sx;
        public int sy;
        public string layer_type;
        public int stride;
        public float l1_decay_mul;
        public float l2_decay_mul;
        public int pad;
        public MyConvNetJSWeightLayerVol[] filters;
        public MyConvNetJSWeightLayerVol biases;
        public int num_inputs;

    }

    public class MyConvNetJSWeightLayerVol
    {
        public int sx;
        public int sy;
        public int depth;
        public Dictionary<string, double> w;
    }
}
