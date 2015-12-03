using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace GoodAI.BrainSimulator.Nodes
{
    public class NodeToolStripRenderer : ToolStripProfessionalRenderer
    {
        protected override void OnRenderArrow(ToolStripArrowRenderEventArgs e) {
            e.Direction = ArrowDirection.Right;
            base.OnRenderArrow(e);
        }
    }
}
