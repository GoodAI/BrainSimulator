using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace GoodAI.BrainSimulator.Utils
{
    /// <summary>
    /// ToolStripTextBox with cue banner support. Borrowed from
    /// http://www.aaronlerch.com/blog/2007/12/01/watermarked-edit-controls/
    /// </summary>
    public class CueToolStripTextBox : ToolStripTextBox
    {
        public CueToolStripTextBox()
            : base()
        {
            if (this.Control != null)
            {
                this.Control.HandleCreated += new EventHandler(OnControlHandleCreated);
            }
        }

        public CueToolStripTextBox(string name)
            : base(name)
        {
            if (this.Control != null)
            {
                this.Control.HandleCreated += new EventHandler(OnControlHandleCreated);
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (this.Control != null)
                {
                    this.Control.HandleCreated -= new EventHandler(OnControlHandleCreated);
                }
            }

            base.Dispose(disposing);
        }

        void OnControlHandleCreated(object sender, EventArgs e)
        {
            UpdateCue();
        }

        #region PInvoke Helpers

        private static uint ECM_FIRST = 0x1500;
        private static uint EM_SETCUEBANNER = ECM_FIRST + 1;

        [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = false)]
        private static extern IntPtr SendMessage(HandleRef hWnd, uint Msg, IntPtr wParam, String lParam);

        #endregion PInvoke Helpers

        #region CueText

        private string _cueText = String.Empty;

        /// <summary>
        /// Gets or sets the text the <see cref="TextBox"/> will display as a cue to the user.
        /// </summary>
        [Description("The text value to be displayed as a cue to the user.")]
        [Category("Appearance")]
        [DefaultValue("")]
        [Localizable(true)]
        public string CueText
        {
            get { return _cueText; }
            set
            {
                if (value == null)
                {
                    value = String.Empty;
                }

                if (!_cueText.Equals(value, StringComparison.CurrentCulture))
                {
                    _cueText = value;
                    UpdateCue();
                    OnCueTextChanged(EventArgs.Empty);
                }
            }
        }

        /// <summary>
        /// Occurs when the <see cref="CueText"/> property value changes.
        /// </summary>
        public event EventHandler CueTextChanged;

        [EditorBrowsable(EditorBrowsableState.Advanced)]
        protected virtual void OnCueTextChanged(EventArgs e)
        {
            EventHandler handler = CueTextChanged;
            if (handler != null)
            {
                handler(this, e);
            }
        }

        #endregion CueText

        #region ShowCueTextOnFocus

        private bool _showCueTextWithFocus = false;

        /// <summary>
        /// Gets or sets a value indicating whether the <see cref="TextBox"/> will display the <see cref="CueText"/>
        /// even when the control has focus.
        /// </summary>
        [Description("Indicates whether the CueText will be displayed even when the control has focus.")]
        [Category("Appearance")]
        [DefaultValue(false)]
        [Localizable(true)]
        public bool ShowCueTextWithFocus
        {
            get { return _showCueTextWithFocus; }
            set
            {
                if (_showCueTextWithFocus != value)
                {
                    _showCueTextWithFocus = value;
                    UpdateCue();
                    OnShowCueTextWithFocusChanged(EventArgs.Empty);
                }
            }
        }

        /// <summary>
        /// Occurs when the <see cref="ShowCueTextWithFocus"/> property value changes.
        /// </summary>
        public event EventHandler ShowCueTextWithFocusChanged;

        [EditorBrowsable(EditorBrowsableState.Advanced)]
        protected virtual void OnShowCueTextWithFocusChanged(EventArgs e)
        {
            EventHandler handler = ShowCueTextWithFocusChanged;
            if (handler != null)
            {
                handler(this, e);
            }
        }

        #endregion ShowCueTextOnFocus

        private void UpdateCue()
        {
            // If the handle isn't yet created, 
            // this will be called when it is created
            if ((this.Control != null) && (this.Control.IsHandleCreated))
            {
                SendMessage(new HandleRef(this.Control, this.Control.Handle), EM_SETCUEBANNER, (_showCueTextWithFocus) ? new IntPtr(1) : IntPtr.Zero, _cueText);
            }
        }
    }
}
