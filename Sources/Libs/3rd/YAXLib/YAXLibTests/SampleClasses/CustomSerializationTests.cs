using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;
using System.Xml.Linq;

namespace YAXLibTests.SampleClasses
{
    [YAXCustomSerializer(typeof(CustomMessageSerializer))]
    public class Message
    {
        public string MessageText { get; set; }
        public string BoldContent { get; set; }
        public int BoldIndex { get; set; }
        public int BoldLength { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }
    }

    [ShowInDemoApplication]
    public class CustomSerializationTests
    {
        [YAXCustomSerializer(typeof(CustomTitleSerializer))]
        [YAXElementFor("SomeTitle")]
        public string Title { get; set; }

        public Message Message { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static CustomSerializationTests GetSampleInstance()
        {
            var message = new Message
            {
                MessageText = "The sun is shining",
                BoldContent = "sun",
                BoldIndex = 4,
                BoldLength = 3
            };

            return new CustomSerializationTests()
            {
                Title = "Important Note",
                Message = message,
            };
        }
    }

    public class CustomMessageSerializer : ICustomSerializer<Message>
    {
        #region ICustomSerializer<Message> Members

        public void SerializeToAttribute(Message objectToSerialize, XAttribute attrToFill)
        {
            throw new NotImplementedException();
        }

        public void SerializeToElement(Message objectToSerialize, XElement elemToFill)
        {
            string message = objectToSerialize.MessageText;
            string beforeBold = message.Substring(0, objectToSerialize.BoldIndex);
            string afterBold = message.Substring(objectToSerialize.BoldIndex + objectToSerialize.BoldLength);

            elemToFill.Add(new XText(beforeBold));
            elemToFill.Add(new XElement("b", objectToSerialize.BoldContent));
            elemToFill.Add(new XText(afterBold));
        }

        public string SerializeToValue(Message objectToSerialize)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region ICustomDeserializer<Message> Members

        public Message DeserializeFromAttribute(XAttribute attrib)
        {
            throw new NotImplementedException();
        }

        public Message DeserializeFromElement(XElement element)
        {
            string wholeMessage = "";
            int boldIndex = -1;
            int boldLength = -1;
            string boldContent = "";

            foreach (XNode node in element.Nodes())
            {
                if (node is XText)
                {
                    wholeMessage += (node as XText).Value;
                }
                else if (node is XElement)
                {
                    XElement boldElement = node as XElement;
                    if (boldElement.Name.ToString() == "b")
                    {
                        boldContent = boldElement.Value;
                        boldIndex = wholeMessage.Length;
                        boldLength = boldContent.Length;
                        wholeMessage += boldContent;
                    }
                }
            }

            return new Message()
            {
                MessageText = wholeMessage,
                BoldContent = boldContent,
                BoldIndex = boldIndex,
                BoldLength = boldLength
            };
        }

        public Message DeserializeFromValue(string value)
        {
            throw new NotImplementedException();
        }

        #endregion

    }

    public class CustomTitleSerializer : ICustomSerializer<string>
    {
        #region ICustomDeserializer<string> Members

        public string DeserializeFromAttribute(XAttribute attrib)
        {
            return RetreiveValue(attrib.Value);
        }

        public string DeserializeFromElement(XElement element)
        {
            return RetreiveValue(element.Value);
        }

        public string DeserializeFromValue(string value)
        {
            return RetreiveValue(value);
        }

        private string RetreiveValue(string str)
        {
            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < str.Length; i++)
            {
                if (i % 2 == 0)
                    sb.Append(str[i]);
            }

            return sb.ToString();
        }

        #endregion

        #region ICustomSerializer<string> Members

        public void SerializeToAttribute(string objectToSerialize, XAttribute attrToFill)
        {
            attrToFill.Value = CreateMixedValue(objectToSerialize);
        }

        public void SerializeToElement(string objectToSerialize, XElement elemToFill)
        {
            elemToFill.Value = CreateMixedValue(objectToSerialize);
        }

        public string SerializeToValue(string objectToSerialize)
        {
            return CreateMixedValue(objectToSerialize);
        }

        private string CreateMixedValue(string str)
        {
            StringBuilder sb = new StringBuilder(2 * str.Length);

            for (int i = 0; i < str.Length; i++)
            {
                sb.Append(str[i] + ".");
            }

            return sb.ToString();
        }

        #endregion
    }
}
