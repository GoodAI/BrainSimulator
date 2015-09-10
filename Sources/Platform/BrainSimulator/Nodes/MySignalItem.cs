using GoodAI.Core.Signals;
using Graph;
using Graph.Items;
using System;
using System.Drawing;

namespace GoodAI.BrainSimulator.Nodes
{
    internal class MySignalItem : NodeLabelItem
    {
        protected MySignal m_signal;

        private const float SIGNAL_SIZE = 8f;
        private const  float SIGNAL_OFFSET = 3.5f;


        public MySignalItem(MySignal signal) :
			base(signal.Name, false, false)
		{
            m_signal = signal;            
		}

        internal override SizeF Measure(Graphics graphics)
        {
            if (m_signal is MyProxySignal)
            {
                if ((m_signal as MyProxySignal).Source != null)
                {
                    this.Text = (m_signal as MyProxySignal).Source.Name;
                }
                else
                {
                    this.Text = "<none>";
                }
            }
            else
            {
                this.Text = m_signal.Name;
            }

            SizeF size = base.Measure(graphics);
            return new SizeF(size.Width + 2 * (SIGNAL_SIZE + SIGNAL_OFFSET), size.Height);
        }

        internal override void Render(Graphics graphics, SizeF minimumSize, PointF location)
        {         
            var size = Measure(graphics);
            size.Width = Math.Max(minimumSize.Width, size.Width);
            size.Height = Math.Max(minimumSize.Height, size.Height);            
            
            Brush incomingSignalColor = Brushes.LightSteelBlue;
            Brush ownSignalColor = Brushes.LightSteelBlue;            

            if (m_signal.IsIncomingRised())
            {
                incomingSignalColor = Brushes.LimeGreen;
            }

            if (m_signal.IsDropped())
            {
                ownSignalColor = Brushes.Tomato;
            }
            if (m_signal.IsRised())
            {
                ownSignalColor = Brushes.LimeGreen;
            }

            graphics.FillEllipse(incomingSignalColor, location.X + SIGNAL_OFFSET, location.Y + SIGNAL_OFFSET, SIGNAL_SIZE, SIGNAL_SIZE);
            graphics.DrawEllipse(Pens.DimGray, location.X + SIGNAL_OFFSET, location.Y + SIGNAL_OFFSET, SIGNAL_SIZE, SIGNAL_SIZE);

            graphics.FillEllipse(ownSignalColor, location.X + 2 * SIGNAL_OFFSET + SIGNAL_SIZE, location.Y + SIGNAL_OFFSET, SIGNAL_SIZE, SIGNAL_SIZE);
            graphics.DrawEllipse(Pens.DimGray, location.X + 2 * SIGNAL_OFFSET + SIGNAL_SIZE, location.Y + SIGNAL_OFFSET, SIGNAL_SIZE, SIGNAL_SIZE);

            graphics.DrawString(this.Text, SystemFonts.MenuFont, Brushes.Black, new RectangleF(location, size), GraphConstants.RightTextStringFormat);
        }
    }
}
