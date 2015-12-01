using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using GoodAI.Core.Configuration;

namespace GoodAI.BrainSimulator.Nodes
{
    internal class UiNodeInfo
    {
        public UiNodeInfo(ListViewItem listViewItem, MyNodeConfig config,  string searchableText)
        {
            ListViewItem = listViewItem;
            Config = config;
            SearchableText = searchableText;
        }

        public ListViewItem ListViewItem { get; private set; }

        public MyNodeConfig Config { get; private set; }

        public string SearchableText { get; private set; }

        public bool Matches(string phrase)
        {
            // do case insensitive search
            return SearchableText.IndexOf(phrase, StringComparison.OrdinalIgnoreCase) >= 0;
        }
    }
}
