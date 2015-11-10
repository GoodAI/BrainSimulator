using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Configuration;

namespace GoodAI.BrainSimulator.Nodes
{
    public class CategorySortingHat  // TODO: make internal
    {
        private ISet<NodeCategory> m_categories = new HashSet<NodeCategory>();

        private IList<MyNodeConfig> m_nodes = new List<MyNodeConfig>(100);

        internal IEnumerable<NodeCategory> Categories
        {
            get { return m_categories; }
        }

        public IEnumerable<MyNodeConfig> Nodes
        {
            get { return m_nodes; }
        }

        public static string DetectCategoryName(MyNodeConfig nodeConfig)
        {
            string categoryName;

            if (nodeConfig.Labels != null && nodeConfig.Labels.Count > 0)
            {
                categoryName = nodeConfig.Labels[0];
            }
            else  // Fallback when there are no labels
            {
                string nameSpace = nodeConfig.NodeType.Namespace;
                
                if (nameSpace.StartsWith("GoodAI.Modules."))  // TODO: consider using whenever namespace level >= 3
                {
                    categoryName = nameSpace.Split(new char[] { '.' })[2];  // take the third namespace level
                }
                else if (nameSpace.LastIndexOf('.') > 0)
                { 
                    categoryName = nameSpace.Substring(0, nameSpace.LastIndexOf('.'));  // strip the last level
                }
                else
                {
                    categoryName = nameSpace;
                }
            }

            return categoryName;
        }

        public void AddNodeAndCategory(MyNodeConfig nodeConfig)
        {
            m_nodes.Add(nodeConfig);  // TODO(Premek): adding nodes is not needed on some places
            
            m_categories.Add(GetNodeCategory(nodeConfig));
        }

        private NodeCategory GetNodeCategory(MyNodeConfig nodeConfig)
        {
            string categoryName = DetectCategoryName(nodeConfig);

            NodeCategory nodeCategory = (MyConfiguration.KnownCategories.ContainsKey(categoryName))
                ? new NodeCategory(MyConfiguration.KnownCategories[categoryName])
                : new NodeCategory(categoryName);

            return nodeCategory;
        }
    }
}
