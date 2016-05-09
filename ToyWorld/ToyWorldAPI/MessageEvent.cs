using System;

namespace GoodAI.ToyWorldAPI
{
    public interface IMessageSender
    {
        /// <summary>
        /// Message obtained from avatar
        /// </summary>
        event MessageEventHandler NewMessage;
    }

    public class MessageEventArgs : EventArgs
    {
        public string Message { get; set; }

        public MessageEventArgs(string message)
        {
            Message = message;
        }
    }

    public delegate void MessageEventHandler(object sender, MessageEventArgs e);
}
