using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using GoodAI.BrainSimulator.Properties;
using GoodAI.Core.Configuration;

namespace GoodAI.BrainSimulator.Nodes
{
    internal class NodeCategory
    {
        private MyCategoryConfig m_config;

        public NodeCategory(string name)
        {
            Name = name;
        }

        public NodeCategory(MyCategoryConfig config)
        {
            m_config = config;

            Name = config.Name;
        }

        public override int GetHashCode()
        {
            return Name.GetHashCode();
        }
        
        // TODO(Premek): also define NodeCategory.Equals for greater performance
        public override bool Equals(object obj)
        {
            if (obj == null)
                return false;

            var nodeCategory = obj as NodeCategory;
            if (nodeCategory == null)
                return false;

            return Name.Equals(nodeCategory.Name);
        }

        public string Name { get; private set; }

        public Image SmallImage
        {
            get
            {
                if (m_config == null)
                {
                    return (Name == "Worlds") ? Resources.world : Resources.keypad;
                }

                return m_config.SmallImage;
            }
        }
    }
}
