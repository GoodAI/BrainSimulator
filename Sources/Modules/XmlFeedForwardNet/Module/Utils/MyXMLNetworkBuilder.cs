using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.IO;
using XmlFeedForwardNet.Layers;
using BrainSimulator.Utils;
using XmlFeedForwardNet.Utils.WeightLoader;
using System.Globalization;
using XmlFeedForwardNet.Networks;
using XmlFeedForwardNet.Layers.Mirror;

namespace XmlFeedForwardNet.Utils
{
    class MyXMLNetworkBuilder
    {

        public class MyXMLBuilderException : Exception
        {
            public MyXMLBuilderException(string msg) : base(msg) { }
        }

        private MyXMLNetNode m_network;
        private Dictionary<string, MyAbstractFLayer> m_layerIds;
        private MyConvNetJSWeightLoader m_weightLoader = null;
        private uint m_convolutionLayerCount = 0;
        private uint m_neuronLayerCount = 0;
        private int m_inputWidth;
        private int m_inputHeight;
        private int m_inputCount;
        private int m_ForwardSamplesPerStep;
        private int m_TrainingSamplesPerStep;

        public MyXMLNetworkBuilder(MyXMLNetNode network)
        {
            m_network = network;
            m_layerIds = new Dictionary<string, MyAbstractFLayer>();
            m_convolutionLayerCount = 0;
            m_neuronLayerCount = 0;

            // Default values
            m_network.Output.MinValueHint = -1;
            m_network.Output.MaxValueHint = 1;
        }

        public void Build(string buildFilePath)
        {
            bool networkMissing = true;
            StreamReader streamReader;
            string xmlContent;
            try
            {
                streamReader = new StreamReader(buildFilePath);
                xmlContent = streamReader.ReadToEnd();
                streamReader.Close();
            }
            catch (IOException e)
            {
                //throw new MyXMLBuilderException(e.Message + " \"" + buildFilePath + " \"");
                throw new MyXMLBuilderException(e.Message);
            }

            xmlContent = xmlContent.ToLowerInvariant();

            XmlDocument xmlDoc = new XmlDocument();
            xmlDoc.LoadXml(xmlContent);
            foreach (XmlNode docItem in xmlDoc.ChildNodes)
            {
                if (docItem.NodeType == XmlNodeType.Element)
                {
                    switch (docItem.Name)
                    {
                        case "network":
                            BuildNetwork(docItem);
                            networkMissing = false;
                            break;
                        default:
                            throw new MyXMLBuilderException("Unknown tag: " + docItem.Name);

                    }
                }
            }

            if (networkMissing)
                throw new MyXMLBuilderException("<network> tag not found");

        }


        private void BuildNetwork(XmlNode networkItem)
        {
            // Set the global settings
            BuildNetworkOutput(networkItem);


            m_network.InputWidth = (uint)m_inputWidth;
            m_network.InputHeight = (uint)m_inputHeight;
            m_network.InputsCount = (uint)m_inputCount;
            m_network.ForwardSamplesPerStep = (uint)m_ForwardSamplesPerStep;
            if (m_TrainingSamplesPerStep > 0)
            {
                m_network.TrainingSamplesPerStep = (uint)m_TrainingSamplesPerStep;
            }
            else
            {
                m_network.TrainingSamplesPerStep = (uint)m_ForwardSamplesPerStep;
            }

            // Build input layer
            m_network.InputLayer = new MyInputLayer(m_network, m_network.DataInput, 0, m_network.InputsCount, m_network.InputWidth, m_network.InputHeight, m_network.ForwardSamplesPerStep);


            // Build layers
            BuildNetworkLayer(networkItem);
        }

        private bool GetBooleanValue(string str)
        {
            if (str == "yes" || str == "true")
                return true;
            else if (str == "no" || str == "false")
                return false;
            else
                throw new MyXMLBuilderException("Incorrect boolean value. Correct values are 'yes', 'true', 'no', 'false'.");
        }

        private void BuildNetworkOutput(XmlNode networkItem)
        {
            foreach (XmlAttribute networkAttribute in networkItem.Attributes)
            {
                switch (networkAttribute.Name)
                {
                    case "inputwidth":
                        m_inputWidth = int.Parse(networkAttribute.Value);
                        break;
                    case "inputheight":
                        m_inputHeight = int.Parse(networkAttribute.Value);
                        break;
                    case "inputcount":
                        m_inputCount = int.Parse(networkAttribute.Value);
                        if (m_inputCount <= 0)
                            throw new MyXMLBuilderException("inputCount must be positive");
                        break;
                    case "samplesperstep":
                        m_ForwardSamplesPerStep = int.Parse(networkAttribute.Value);
                        if (m_ForwardSamplesPerStep <= 0)
                            throw new MyXMLBuilderException("SamplesPerStep must be positive");
                        break;
                    case "trainingsamplesperstep":
                        m_TrainingSamplesPerStep = int.Parse(networkAttribute.Value);
                        if (m_TrainingSamplesPerStep <= 0)
                            throw new MyXMLBuilderException("TrainingSamplesPerStep must be positive");
                        break;
                    case "outputminvalue":
                        m_network.Output.MinValueHint = float.Parse(networkAttribute.Value, CultureInfo.InvariantCulture);
                        break;
                    case "outputmaxvalue":
                        m_network.Output.MaxValueHint = float.Parse(networkAttribute.Value, CultureInfo.InvariantCulture);
                        break;
                    case "outputcolumnhint":
                        m_network.Output.ColumnHint = int.Parse(networkAttribute.Value);
                        if (m_network is MyXMLNetNode)
                            (m_network as MyXMLNetNode).XMLColumnHint = int.Parse(networkAttribute.Value);
                        break;
                    case "useweightfile":
                        if (GetBooleanValue(networkAttribute.Value))
                            if (networkItem.Attributes.GetNamedItem("weightfilepath") == null)
                                throw new MyXMLBuilderException("useWeightFile set to true, but weight file path missing. Use attribute weightFilePath=\"path\"");
                        break;
                    case "weightfilepath":
                        XmlNode useweightfileAttr = networkItem.Attributes["useweightfile"];
                        if ((useweightfileAttr == null) || GetBooleanValue(useweightfileAttr.Value))
                        {
                            XmlNode weightFileFormatNode = networkItem.Attributes.GetNamedItem("weightfileformat");
                            if (weightFileFormatNode == null)
                                throw new MyXMLBuilderException("Weight file format missing. Use attribute weightFileFormat=\"my_format\"");
                            if (weightFileFormatNode.Value == "convnetjs")
                            {
                                try
                                {
                                    m_weightLoader = new MyConvNetJSWeightLoader(networkAttribute.Value);
                                }
                                catch (Exception e)
                                {
                                    throw new MyXMLBuilderException("Incorrect value for weightfilepath: " + e.Message);
                                }
                            }
                            else
                                throw new MyXMLBuilderException("Unknown weight file format: " + weightFileFormatNode.Value);
                        }
                        break;
                    case "weightfileformat":
                        // Treated elsewhere
                        break;
                    default:
                        throw new MyXMLBuilderException("Unknown attribute: " + networkAttribute.Name);
                }
            }

            if (m_network.Output.MinValueHint > m_network.Output.MaxValueHint)
                throw new MyXMLBuilderException("MaxValue must be greater than MinValue");

            if (m_network.Output.ColumnHint < 0)
                throw new MyXMLBuilderException("ColumnHint must be positive");



        }


        private void BuildNetworkLayer(XmlNode networkItem)
        {
            bool featureLayerFound = false;
            foreach (XmlNode layerItem in networkItem.ChildNodes)
            {
                if (layerItem.NodeType == XmlNodeType.Element)
                {
                    try
                    {
                        string layerId = null;

                        // Get the id if exist
                        foreach (XmlAttribute layerAttribute in layerItem.Attributes)
                            if (layerAttribute.Name == "id")
                            {
                                layerId = layerAttribute.Value;
                                if (m_layerIds.ContainsKey(layerId))
                                    throw new MyXMLBuilderException("Another layer already has the id \"" + layerId + "\"");

                                layerItem.Attributes.Remove(layerAttribute);
                                break;
                            }

                        MyAbstractFLayer createdLayer = null;
                        switch (layerItem.Name)
                        {
                            case "convolutionlayer":
                                createdLayer = BuildNetworkLayerConvolution(layerItem);
                                break;
                            case "poollayer":
                                createdLayer = BuildNetworkLayerPool(layerItem);
                                break;
                            case "neuronlayer":
                                createdLayer = BuildNetworkLayerNeuron(layerItem);
                                break;
                            case "neuroncopylayer":
                                createdLayer = BuildNetworkLayerNeuronCopy(layerItem);
                                break;
                            case "linearlayer":
                                createdLayer = BuildNetworkLayerLinear(layerItem);
                                break;
                            case "softmaxlayer":
                                createdLayer = BuildNetworkLayerSoftmax(layerItem);
                                break;
                            case "activationlayer":
                                createdLayer = BuildNetworkLayerActivation(layerItem);
                                break;
                            case "mirrorconvolutionlayer":
                                createdLayer = BuildNetworkLayerMirrorConvolution(layerItem);
                                break;
                            case "mirrorpoollayer":
                                createdLayer = BuildNetworkLayerMirrorPool(layerItem);
                                break;
                            case "mirrorneuronlayer":
                                createdLayer = BuildNetworkLayerMirrorNeuron(layerItem);
                                break;
                            case "featurelayer":
                                if (featureLayerFound)
                                    throw new MyXMLBuilderException("The featureLayer must be unique in a network");
                                else if (m_network.Layers.Count == 0)
                                    throw new MyXMLBuilderException("The featureLayer cannot be placed as fist layer");
                                else
                                    m_network.FeatureLayerPosition = m_network.Layers.Count - 1;
                                break;
                            case "mirrorlinearlayer":
                                throw new MyXMLBuilderException("Layer " + layerItem.Name + " not implemented");
                            default:
                                throw new MyXMLBuilderException("Unknown layer: " + layerItem.Name);
                        }

                        // Store the created layer
                        if (layerId != null && createdLayer != null)
                            m_layerIds[layerId] = createdLayer;

                    }
                    catch (MyXMLBuilderException e)
                    {
                        throw new MyXMLBuilderException(layerItem.Name + ": " + e.Message);
                    }
                }
            }
        }



        /*
         * LAYERS
         */

        private L GetLayerById<L>(string id) where L : MyAbstractFLayer
        {
            // Check if the referenced layer exists
            if (m_layerIds.ContainsKey(id))
            {
                MyAbstractFLayer abstractLayer = m_layerIds[id];
                // Check if the referenced layer is the right type
                if (abstractLayer is L)
                    return abstractLayer as L;
                else
                    throw new MyXMLBuilderException("Original layer must be of type " + typeof(L).AssemblyQualifiedName);
            }
            else
                throw new MyXMLBuilderException("Unknown layer id \"" + id + "\"");
        }



        private MyAbstractWeightLayer BuildNetworkLayerConvolution(XmlNode layerItem)
        {
            uint featuresCount = 0;
            uint patchWidth = 0;
            uint patchHeight = 0;
            uint xStride = 1;
            uint yStride = 1;

            bool featuresCountMissing = true;
            bool patchWidthMissing = true;
            bool patchHeightMissing = true;

            // Parse attributes
            foreach (XmlAttribute layerAttribute in layerItem.Attributes)
            {
                switch (layerAttribute.Name)
                {
                    case "nbfeatures":
                        featuresCount = uint.Parse(layerAttribute.Value);
                        featuresCountMissing = false;
                        break;
                    case "patchwidth":
                        patchWidth = uint.Parse(layerAttribute.Value);
                        patchWidthMissing = false;
                        break;
                    case "patchheight":
                        patchHeight = uint.Parse(layerAttribute.Value);
                        patchHeightMissing = false;
                        break;
                    case "patchsize":
                        patchWidth = uint.Parse(layerAttribute.Value);
                        patchHeight = patchWidth;
                        patchHeightMissing = false;
                        patchWidthMissing = false;
                        break;
                    case "stride":
                        xStride = uint.Parse(layerAttribute.Value);
                        yStride = xStride;
                        break;
                    case "xstride":
                        xStride = uint.Parse(layerAttribute.Value);
                        break;
                    case "ystride":
                        yStride = uint.Parse(layerAttribute.Value);
                        break;
                    default:
                        throw new MyXMLBuilderException("Unknown attribute: " + layerAttribute.Name);
                }
            }

            List<List<uint>> featureMaps = new List<List<uint>>();

            // Parse feature map input indexes
            foreach (XmlNode featureMapItem in layerItem.ChildNodes)
            {
                if (featureMapItem.NodeType == XmlNodeType.Element)
                    switch (featureMapItem.Name)
                    {
                        case "featuremap":
                            List<uint> inputIndexes = new List<uint>();
                            foreach (XmlAttribute featureMapAttribute in featureMapItem.Attributes)
                            {
                                switch (featureMapAttribute.Name)
                                {
                                    case "inputindex":
                                        string[] indexStrTab = featureMapAttribute.Value.Split(' ');
                                        foreach (string valueStr in indexStrTab)
                                        {
                                            int value = int.Parse(valueStr);
                                            if (value < 0)
                                                throw new MyXMLBuilderException("inputIndex must be null or positive");
                                            inputIndexes.Add((uint)value);
                                        }
                                        break;
                                    default:
                                        throw new MyXMLBuilderException("Unknown attribute: " + featureMapAttribute.Name);
                                }
                            }

                            if (inputIndexes.Count == 0)
                                throw new MyXMLBuilderException("Missing indexes in featureMap attribute");

                            featureMaps.Add(inputIndexes);
                            featuresCountMissing = false;
                            break;
                        default:
                            throw new MyXMLBuilderException("Unknown tag: " + featureMapItem.Name);
                    }
            }

            uint[][] featureMapsArray = null;
            if (featureMaps.Count > 0)
            {
                featuresCount = (uint)featureMaps.Count;
                featureMapsArray = new uint[featureMaps.Count][];
                for (int i = 0; i < featureMaps.Count; i++)
                {
                    int nbInput = featureMaps[i].Count;
                    featureMapsArray[i] = new uint[nbInput];
                    for (int j = 0; j < nbInput; j++)
                        featureMapsArray[i][j] = featureMaps[i][j];
                }
            }

            // Validate values
            if (featuresCountMissing)
                throw new MyXMLBuilderException("Missing nbFeatures parameter");
            if (patchWidthMissing)
                throw new MyXMLBuilderException("Missing patchWidth parameter");
            if (patchHeightMissing)
                throw new MyXMLBuilderException("Missing patchHeight parameter");

            if (featuresCount <= 0)
                throw new MyXMLBuilderException("Must have a positive number of features");
            if (patchWidth <= 0)
                throw new MyXMLBuilderException("Patch width must be positive");
            if (patchHeight <= 0)
                throw new MyXMLBuilderException("Patch height must be positive");

            // Success!
            float[] initialWeight = null;
            float[] initialBias = null;
            if (m_weightLoader != null)
            {
                initialWeight = m_weightLoader.ConvWeight[(int)m_convolutionLayerCount];
                initialBias = m_weightLoader.ConvBias[(int)m_convolutionLayerCount];
            }
            MyAbstractWeightLayer layer = new MyConvolutionLayer(m_network, featuresCount, patchWidth, patchHeight, xStride, yStride, featureMapsArray, initialWeight, initialBias);
            m_network.AddLayer(layer);
            m_convolutionLayerCount++;
            return layer;
        }



        private MyAbstractFBLayer BuildNetworkLayerPool(XmlNode layerItem)
        {
            uint stride = 0;
            MyPoolLayer.MyPoolRule rule = MyPoolLayer.MyPoolRule.MAX;

            bool strideMissing = true;
            bool ruleMissing = true;

            // Parse
            foreach (XmlAttribute layerAttribute in layerItem.Attributes)
            {
                switch (layerAttribute.Name)
                {
                    case "stride":
                        stride = uint.Parse(layerAttribute.Value);
                        strideMissing = false;
                        break;
                    case "rule":
                        switch (layerAttribute.Value)
                        {
                            case "max":
                                rule = MyPoolLayer.MyPoolRule.MAX;
                                ruleMissing = false;
                                break;
                            case "average":
                                rule = MyPoolLayer.MyPoolRule.AVERAGE;
                                ruleMissing = false;
                                break;
                            default:
                                throw new MyXMLBuilderException("Unknown pooling rule: " + layerAttribute.Value);
                        }
                        break;
                    default:
                        throw new MyXMLBuilderException("Unknown attribute: " + layerAttribute.Name);
                }
            }

            // Validate values
            if (strideMissing)
                throw new MyXMLBuilderException("Missing stride parameter");
            if (ruleMissing)
                throw new MyXMLBuilderException("Missing rule parameter");

            if (stride <= 0)
                throw new MyXMLBuilderException("Stride must be positive");

            // Success!
            MyAbstractFBLayer layer = new MyPoolLayer(m_network, stride, rule);
            m_network.AddLayer(layer);
            return layer;
        }



        private MyAbstractWeightLayer BuildNetworkLayerNeuron(XmlNode layerItem)
        {
            uint nbNeurons = 0;

            bool nbNeuronsMissing = true;

            // Parse
            foreach (XmlAttribute layerAttribute in layerItem.Attributes)
            {
                switch (layerAttribute.Name)
                {
                    case "nbneurons":
                        nbNeurons = uint.Parse(layerAttribute.Value);
                        nbNeuronsMissing = false;
                        break;
                    default:
                        throw new MyXMLBuilderException("Unknown attribute: " + layerAttribute.Name);
                }
            }

            // Validate values
            if (nbNeuronsMissing)
                throw new MyXMLBuilderException("Missing nbNeurons parameter");

            if (nbNeurons == 0)
                throw new MyXMLBuilderException("Must have at least 1 neuron");

            // Success!
            float[] initialWeight = null;
            float[] initialBias = null;
            if (m_weightLoader != null)
            {
                initialWeight = m_weightLoader.NeuronWeight[(int)m_neuronLayerCount];
                initialBias = m_weightLoader.NeuronBias[(int)m_neuronLayerCount];
            }
            MyAbstractWeightLayer layer = new MyNeuronLayer(m_network, nbNeurons, initialWeight, initialBias);
            m_network.AddLayer(layer);
            m_neuronLayerCount++;
            return layer;
        }

        private MyAbstractWeightLayer BuildNetworkLayerNeuronCopy(XmlNode layerItem)
        {
            uint nbNeurons = 0;

            bool nbNeuronsMissing = true;

            // Parse
            foreach (XmlAttribute layerAttribute in layerItem.Attributes)
            {
                switch (layerAttribute.Name)
                {
                    case "nbneurons":
                        nbNeurons = uint.Parse(layerAttribute.Value);
                        nbNeuronsMissing = false;
                        break;
                    default:
                        throw new MyXMLBuilderException("Unknown attribute: " + layerAttribute.Name);
                }
            }

            // Validate values
            if (nbNeuronsMissing)
                throw new MyXMLBuilderException("Missing nbNeurons parameter");

            if (nbNeurons == 0)
                throw new MyXMLBuilderException("Must have at least 1 neuron");

            // Success!
            float[] initialWeight = null;
            float[] initialBias = null;
            MyAbstractWeightLayer layer = new MyNeuronCopyLayer(m_network, nbNeurons, initialWeight, initialBias);
            m_network.AddLayer(layer);
            return layer;
        }

        private MyAbstractWeightLayer BuildNetworkLayerLinear(XmlNode layerItem)
        {
            // Parse
            foreach (XmlAttribute layerAttribute in layerItem.Attributes)
            {
                switch (layerAttribute.Name)
                {
                    default:
                        throw new MyXMLBuilderException("Unknown attribute: " + layerAttribute.Name);
                }
            }

            // Success!
            float[] initialWeight = null;
            float[] initialBias = null;
            if (m_weightLoader != null)
            {
                initialWeight = m_weightLoader.LinearWeight[(int)m_neuronLayerCount];
                initialBias = m_weightLoader.LinearBias[(int)m_neuronLayerCount];
            }
            MyAbstractWeightLayer layer = new MyLinearLayer(m_network, initialWeight, initialBias);
            m_network.AddLayer(layer);
            m_neuronLayerCount++;
            return layer;
        }


        private MyAbstractFBLayer BuildNetworkLayerActivation(XmlNode layerItem)
        {
            MyActivationLayer.MyActivationFunction function = MyActivationLayer.MyActivationFunction.NO_ACTIVATION;

            bool activationMissing = true;

            // Parse
            foreach (XmlAttribute layerAttribute in layerItem.Attributes)
            {
                switch (layerAttribute.Name)
                {
                    case "function":
                        switch (layerAttribute.Value)
                        {
                            case "relu":
                                function = MyActivationLayer.MyActivationFunction.RELU;
                                break;
                            case "logistic":
                                function = MyActivationLayer.MyActivationFunction.LOGISTIC;
                                break;
                            case "tanh":
                                function = MyActivationLayer.MyActivationFunction.TANH;
                                break;
                            case "none":
                                function = MyActivationLayer.MyActivationFunction.NO_ACTIVATION;
                                break;
                            default:
                                throw new MyXMLBuilderException("Unknown activation function: " + layerAttribute.Value);
                        }
                        activationMissing = false;
                        break;
                    default:
                        throw new MyXMLBuilderException("Unknown attribute: " + layerAttribute.Name);
                }
            }

            // Validate values
            if (activationMissing)
                throw new MyXMLBuilderException("Missing activation parameter");

            // Success!

            MyActivationLayer layer = new MyActivationLayer(m_network, function);
            m_network.AddLayer(layer);
            return layer;
        }


        private MyAbstractFBLayer BuildNetworkLayerSoftmax(XmlNode layerItem)
        {
            if (layerItem.Attributes.Count > 0)
                MyLog.WARNING.WriteLine("Softmax layer doesn't take any parameters.");

            // Success!
            MyAbstractFBLayer layer = new MySoftmaxLayer(m_network);
            m_network.AddLayer(layer);
            return layer;
        }



        private MyAbstractWeightLayer BuildNetworkLayerMirrorConvolution(XmlNode layerItem)
        {
            MyConvolutionLayer originalLayer = null;

            bool layerMissing = true;

            // Parse
            foreach (XmlAttribute layerAttribute in layerItem.Attributes)
            {
                switch (layerAttribute.Name)
                {
                    case "original":
                        originalLayer = GetLayerById<MyConvolutionLayer>(layerAttribute.Value);
                        layerMissing = false;
                        break;
                    default:
                        throw new MyXMLBuilderException("Unknown attribute: " + layerAttribute.Name);
                }
            }

            // Validate values
            if (layerMissing)
                throw new MyXMLBuilderException("Missing original layer id");

            // Success!
            MyAbstractWeightLayer layer = new MyMirrorConvolutionLayer(m_network, originalLayer);
            m_network.AddLayer(layer);
            return layer;
        }



        private MyAbstractFBLayer BuildNetworkLayerMirrorPool(XmlNode layerItem)
        {
            MyPoolLayer originalLayer = null;

            bool layerMissing = true;

            // Parse
            foreach (XmlAttribute layerAttribute in layerItem.Attributes)
            {
                switch (layerAttribute.Name)
                {
                    case "original":
                        originalLayer = GetLayerById<MyPoolLayer>(layerAttribute.Value);
                        layerMissing = false;
                        break;
                    default:
                        throw new MyXMLBuilderException("Unknown attribute: " + layerAttribute.Name);
                }
            }

            // Validate values
            if (layerMissing)
                throw new MyXMLBuilderException("Missing original layer id");

            // Success!
            MyAbstractFBLayer layer = new MyMirrorPoolLayer(m_network, originalLayer);
            m_network.AddLayer(layer);
            return layer;
        }


        private MyAbstractWeightLayer BuildNetworkLayerMirrorNeuron(XmlNode layerItem)
        {
            MyNeuronLayer originalLayer = null;

            bool layerMissing = true;

            // Parse
            foreach (XmlAttribute layerAttribute in layerItem.Attributes)
            {
                switch (layerAttribute.Name)
                {
                    case "original":
                        originalLayer = GetLayerById<MyNeuronLayer>(layerAttribute.Value);
                        layerMissing = false;
                        break;
                    default:
                        throw new MyXMLBuilderException("Unknown attribute: " + layerAttribute.Name);
                }
            }

            // Validate values
            if (layerMissing)
                throw new MyXMLBuilderException("Missing original layer id");

            // Success!
            MyAbstractWeightLayer layer = new MyMirrorNeuronLayer(m_network, originalLayer);
            m_network.AddLayer(layer);
            return layer;
        }
    }
}