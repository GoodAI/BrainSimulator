using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Reflection;
using YAXLib;

namespace GoodAI.Core.Configuration
{
    [YAXSerializeAs("Node"), YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AllFields)]
    public class MyNodeConfig
    {
        [YAXAttributeForClass, YAXErrorIfMissed(YAXExceptionTypes.Warning, DefaultValue = false)]
        public bool CanBeAdded { get; set; }

        [YAXAttributeForClass, YAXErrorIfMissed(YAXExceptionTypes.Warning)]
        public string BigIcon { get; set; }

        [YAXAttributeForClass, YAXErrorIfMissed(YAXExceptionTypes.Warning)]
        public string SmallIcon { get; set; }

        [YAXDontSerialize]
        public Image BigImage { get; private set; }

        [YAXDontSerialize]
        public Image SmallImage { get; private set; }

        [YAXAttributeForClass, YAXSerializeAs("type"), YAXErrorIfMissed(YAXExceptionTypes.Error)]
        internal string NodeTypeName { get; set; }

        [YAXDontSerialize]
        public bool IsBasicNode { get; internal set; }

        [YAXDontSerialize]
        public Type NodeType { get; internal set; }

        [YAXCollection(YAXCollectionSerializationTypes.RecursiveWithNoContainingElement, EachElementName = "Observer")]
        internal List<MyObserverConfig> m_observerInfoList = null;

        [YAXCollection(YAXCollectionSerializationTypes.RecursiveWithNoContainingElement, EachElementName = "Label")]
        public List<string> Labels { get; protected set; }

        [YAXDontSerialize]
        public Dictionary<Type, MyObserverConfig> KnownObservers { get; private set; }

        public MyNodeConfig()
        {
            KnownObservers = new Dictionary<Type, MyObserverConfig>();
        }

        public void FinalizeConfig(Assembly assembly)
        {
            NodeType = assembly.GetType(NodeTypeName);

            if (NodeType == null)
            {
                throw new TypeLoadException("Node type not found: " + NodeTypeName);
            }

            InitIcons(assembly);
            AddObservers(assembly); 
        }

        public override string ToString()
        {
            return MyProject.ShortenNodeTypeName(NodeType);
        }

        protected static Image GenerateDefaultImage(Type nodeType, bool bigIcon)
        {
            string typeName = MyProject.ShortenNodeTypeName(nodeType);

            return GenerateDefaultImage(typeName, bigIcon);
        }

        internal static Image GenerateDefaultImage(string typeName, bool bigIcon)
        {
            string back_fileName = bigIcon ? @"plain_big.png" : @"plain.png";
            float x = 0;
            float y = bigIcon ? 6 : 3;
            float size = bigIcon ? 36 : 17;

            string label = "";

            for (int i = 0; i < typeName.Length; i++)
            {
                if (Char.IsUpper(typeName[i]))
                {
                    label += typeName[i];
                    if (label.Length == 2) break;
                }
            }

            if (label.Length < 2)
            {
                label = typeName.Substring(0, 2);
            }

            Image background = MyResources.GetImage(back_fileName);

            Graphics g = Graphics.FromImage(background);
            g.DrawString(label, new Font(new FontFamily("Consolas"), 17, FontStyle.Bold), Brushes.White, x, y);
            return background;
        }

        internal void InitIcons(Assembly assembly)
        {
            if (!String.IsNullOrEmpty(BigIcon))
            {
                BigImage = MyResources.GetImageFromAssembly(assembly, BigIcon);
            }

            if (BigImage == null)
            {
                BigImage = GenerateDefaultImage(NodeType, true);
            }

            if (!String.IsNullOrEmpty(SmallIcon))
            {
                SmallImage = MyResources.GetImageFromAssembly(assembly, SmallIcon);
            }

            if (SmallImage == null)
            {
                SmallImage = GenerateDefaultImage(NodeType, false);
            }
        }

        internal void AddObservers(Assembly assembly)
        {
            if (m_observerInfoList != null)
            {
                foreach (MyObserverConfig oc in m_observerInfoList)
                {
                    oc.ObserverType = assembly.GetType(oc.ObserverTypeName);

                    if (oc.ObserverType != null)
                    {
                        KnownObservers[oc.ObserverType] = oc;   
                    }
                    else
                    {
                        MyLog.ERROR.WriteLine("Observer type not found: " + oc.ObserverTypeName);
                    }                    
                }
            }
            AdoptBaseTypeObservers();
        }

        private void AdoptBaseTypeObservers()
        {
            Type baseType = NodeType.BaseType;

            while (MyConfiguration.KnownNodes.ContainsKey(baseType))
            {
                foreach (MyObserverConfig oc in MyConfiguration.KnownNodes[baseType].KnownObservers.Values)
                {
                    KnownObservers[oc.ObserverType] = oc;
                }

                baseType = baseType.BaseType;
            }
        }
    }

    [YAXSerializeAs("World"), YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AllFields)]
    public class MyWorldConfig : MyNodeConfig { }

    [YAXSerializeAs("Observer"), YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AllFields)]
    public class MyObserverConfig
    {
        [YAXAttributeForClass, YAXSerializeAs("type"), YAXErrorIfMissed(YAXExceptionTypes.Error)]
        internal string ObserverTypeName { get; private set; }

        [YAXDontSerialize]
        public Type ObserverType { get; internal set; }

        public override string ToString()
        {
            return ObserverType.Name;
        }
    }

    [YAXSerializeAs("Category"), YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AllFields)]
    public class MyCategoryConfig
    {
        [YAXAttributeForClass, YAXErrorIfMissed(YAXExceptionTypes.Error)]
        public string Name { get; private set; }

        // TODO(Premek): put to a common ancestor
        [YAXAttributeForClass, YAXErrorIfMissed(YAXExceptionTypes.Warning)]
        public string SmallIcon { get; set; }

        [YAXDontSerialize]
        public Image SmallImage { get; private set; }

        public override string ToString()
        {
            return Name;
        }

        internal void InitIcons(Assembly assembly)
        {
            if (!String.IsNullOrEmpty(SmallIcon))
            {
                SmallImage = MyResources.GetImageFromAssembly(assembly, SmallIcon);
            }

            if (SmallImage == null)
            {
                SmallImage = MyNodeConfig.GenerateDefaultImage(Name, false);
            }
        }
    }
}
