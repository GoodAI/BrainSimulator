using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace Graph.Items
{
    public sealed class AcceptNodeValueChangedEventArgs : CancelEventArgs
    {
        public AcceptNodeValueChangedEventArgs(ValueItemState oldState, ValueItemState newState) { PreviousState = oldState; CurrentState = newState; }
        public AcceptNodeValueChangedEventArgs(ValueItemState oldState, ValueItemState newState, bool cancel) : base(cancel) { PreviousState = oldState; CurrentState = newState; }
        public ValueItemState PreviousState { get; private set; }
        public ValueItemState CurrentState { get; set; }
    }

    public struct ValueItemState
    {
        public string Text;
        public Type Type;

        public ValueItemState(Type type)
        {
            Type = type;
            if (type.IsValueType)
                Text = Activator.CreateInstance(type).ToString();
            else
                Text = "";
        }

        public ValueItemState(string text, Type type)
        {
            Text = text;
            Type = type;
        }

        public override bool Equals(object obj)
        {
            var other = (ValueItemState)obj;
            return Text == other.Text && Type == other.Type;
        }

        public override int GetHashCode()
        {
            return Text.GetHashCode() + Type.GetHashCode();
        }

        public static bool operator ==(ValueItemState c1, ValueItemState c2)
        {
            return c1.Equals(c2);
        }

        public static bool operator !=(ValueItemState c1, ValueItemState c2)
        {
            return !c1.Equals(c2);
        }

        public static object TryParse(ValueItemState state)
        {
            return TryParse(state.Text, state.Type);
        }

        public static object TryParse(string text, Type type)
        {
            var converter = TypeDescriptor.GetConverter(type);
            object res = null;
            try
            {
                res = converter.ConvertFromString(text);
            }
            catch (Exception)
            {
                return null;
            }

            return res;
        }
    }

    public sealed class NodeValueItem : NodeItem
    {
        public event EventHandler<AcceptNodeValueChangedEventArgs> ValueChanged;     

        public NodeValueItem(string text)
            : this(text, typeof(string), false, false)
        {
        }

        public NodeValueItem(Type type)
            : this("", type, false, false)
        {

        }

		public NodeValueItem(string text, Type type)
			: this(text, type, false, false)
		{
		}

        public NodeValueItem(string text, Type type, bool inputEnabled, bool outputEnabled)
            : base(inputEnabled, outputEnabled)
		{
            EnableTypeField = true;
            TextMaxLength = 20;
            if (string.IsNullOrEmpty(text))
                ValueState = new ValueItemState(type);
            else
                ValueState = new ValueItemState(text, type);
            background = Brushes.LightCyan;
		}

		#region Name
		public string Name
		{
			get;
			set;
		}
		#endregion

		#region Text

		string internalText = string.Empty;
		public string Text
		{
            get { return ValueState.Text; }
		}

        public string TextFormat { get; set; }
        public int TextMaxLength { get; set; }
        public string FormattedText
        {
            get 
            {
                if (!string.IsNullOrWhiteSpace(this.Text))
                {
                    if (string.IsNullOrWhiteSpace(TextFormat))
                    {
                        if (Text.Length > TextMaxLength)
                            return Text.Remove(TextMaxLength);
                        else
                            return Text;
                    }
                    else
                    {
                        var output = string.Format(TextFormat, Text, ValueState.Type.Name);
                        if (output.Length > TextMaxLength)
                            return output.Remove(TextMaxLength);
                        else
                            return output;
                    }
                }
                else
                {
                    if (string.IsNullOrEmpty(TextFormat))
                    {
                        return "";
                    }
                    else
                    {
                        var output = string.Format(TextFormat, "<EMPTY>", "");
                        return output;
                    }
                }
            }
        }
		
        #endregion

        public Type ValueType
        {
            get { return ValueState.Type; }
        }

        private ValueItemState m_currentState;
        public ValueItemState ValueState
        {
            get { return m_currentState; }
            set
            {
                if (m_currentState == value)
                    return;
                if (ValueChanged != null)
                {
                    var eventArgs = new AcceptNodeValueChangedEventArgs(m_currentState, value);
                    ValueChanged(this, eventArgs);
                    if (eventArgs.Cancel || ValueItemState.TryParse(eventArgs.CurrentState) == null)
                        return;
                    m_currentState = eventArgs.CurrentState;
                }
                else
                {
                    m_currentState = value;
                }
            }
        }

        public object ParsedValue
        {
            get
            {
                var obj = ValueItemState.TryParse(ValueState);
                return obj;
            }
        }

        public bool EnableTypeField { get; set; }

        public Type[] SupportedTypes { get; set; }


		internal SizeF TextSize;

		public override bool OnDoubleClick()
		{
			base.OnDoubleClick();
            DialogResult result = DialogResult.None;
            if (SupportedTypes.Length == 1 && SupportedTypes[0].IsEnum)
            {
                var form = new SelectionForm();
                form.Text = Name ?? "Select item from list";
                form.Items = SupportedTypes[0].GetEnumNames();
                int index = Array.IndexOf(SupportedTypes[0].GetEnumNames(), ValueState.Text);
                if (index == -1)
                    index = 0;
                form.SelectedIndex = index;
                result = form.ShowDialog();
                if (result == DialogResult.OK)
                {
                    ValueState = new ValueItemState(form.Items[form.SelectedIndex], SupportedTypes[0]);
                }
            }
            else
            {
                var form = new ValueBoxForm();
                form.Text = Name ?? "Value editor";
                form.SetSupportedTypes(SupportedTypes);
                form.InputText = ValueState.Text;
                form.InputType = ValueState.Type;
                form.EnableTypeComboBox = EnableTypeField;
                result = form.ShowDialog();
                if (result == DialogResult.OK && ValueItemState.TryParse(form.InputText, form.InputType) != null)
                {
                    ValueState = new ValueItemState(form.InputText, form.InputType);
                }
            }
			return true;
		}

		internal override SizeF Measure(Graphics graphics)
		{
            if (!string.IsNullOrWhiteSpace(this.FormattedText))
			{
				if (this.TextSize.IsEmpty)
				{
					var size = new Size(GraphConstants.MinimumItemWidth, GraphConstants.MinimumItemHeight);

                    this.TextSize = graphics.MeasureString(this.FormattedText, SystemFonts.MenuFont, size, GraphConstants.LeftMeasureTextStringFormat);
					
					this.TextSize.Width  = Math.Max(size.Width, this.TextSize.Width + 8);
					this.TextSize.Height = Math.Max(size.Height, this.TextSize.Height + 2);
				}
				return this.TextSize;
			} else
			{
				return new SizeF(GraphConstants.MinimumItemWidth, GraphConstants.MinimumItemHeight);
			}
		}

		internal override void Render(Graphics graphics, SizeF minimumSize, PointF location)
		{
			var size = Measure(graphics);
			size.Width  = Math.Max(minimumSize.Width, size.Width);
			size.Height = Math.Max(minimumSize.Height, size.Height);

			var path = GraphRenderer.CreateRoundedRectangle(size, location);
            graphics.FillPath(background, path);

			location.Y += 1;
			location.X += 1;

			if ((state & RenderState.Hover) == RenderState.Hover)
			{
				graphics.DrawPath(Pens.White, path);
                graphics.DrawString(this.FormattedText, SystemFonts.MenuFont, Brushes.Black, new RectangleF(location, size), GraphConstants.LeftTextStringFormat);
			} else
			{
				graphics.DrawPath(Pens.Black, path);
                graphics.DrawString(this.FormattedText, SystemFonts.MenuFont, Brushes.Black, new RectangleF(location, size), GraphConstants.LeftTextStringFormat);
			}
		}
    }
}
