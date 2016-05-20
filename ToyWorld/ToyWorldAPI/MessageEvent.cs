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
        public string Sender { get; set; }

        public MessageEventArgs(string message, string sender = "Unknown")
        {
            Message = message;
            Sender = sender;
        }
    }

    public delegate void MessageEventHandler(object sender, MessageEventArgs e);
}
