using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using System.Security.Cryptography;
using YAXLib;

namespace GoodAI.Modules.Transforms
{

    /// <author>GoodAI</author>
    /// <meta>kk</meta>
    /// <status>Working</status>
    /// <summary>Calculates a hash function of input data.</summary>
    /// <description></description>
    [YAXSerializeAs("Hash")]
    public class MyHash : MyTransform
    {
        public enum MyHashMethod
        {
            MD5Dense,
            MD5Sparse
        }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("Behavior")]
        public MyHashMethod HashMethod { get; set; }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = 1021), YAXElementFor("Behavior")]
        public int SPARSE_SIZE { get; set; }

        public MyMD5Task MD5Task { get; private set; }

        /// <summary>
        /// Calculates standard MD5 hash.
        /// </summary>
        [Description("MD5"), MyTaskInfo(OneShot = false)]
        public class MyMD5Task : MyTask<MyHash>
        {
            private MD5 m_md5;
            private byte[] m_hash;
            private byte[] m_data;

            public override void Init(Int32 nGPU)
            {
                m_md5 = System.Security.Cryptography.MD5.Create();
                m_hash = new byte[16];
                m_data = new byte[Owner.InputSize * sizeof(float)];
            }


            public override void Execute()
            {
                Owner.Input.SafeCopyToHost();
                Buffer.BlockCopy(Owner.Input.Host, 0, m_data, 0, m_data.Length);
                m_hash = m_md5.ComputeHash(m_data);

                if (Owner.HashMethod == MyHashMethod.MD5Dense)
                {
                    //go through all 128 hash bits and set output accordingly
                    int bitPos = 0;
                    while (bitPos < 128)
                    {
                        int byteIndex = bitPos / 8;
                        int offset = bitPos % 8;
                        bool isSet = (m_hash[byteIndex] & (1 << offset)) != 0;

                        // isSet = [True] if the bit at bitPos is set, false otherwise
                        Owner.Output.Host[bitPos] = isSet ? 1.0f : 0.0f;

                        bitPos++;
                    }
                }
                else if (Owner.HashMethod == MyHashMethod.MD5Sparse)
                {
                    for (int i = 0; i < Owner.OutputSize; i++)
                    {
                        Owner.Output.Host[i] = 0.0f;
                    }

                    //copy first 32 bits of byte[] m_hash to int hash
                    uint hash = 0;
                    for (int i = 0; i < 4; i++)
                    {
                        hash = hash | ((uint)(m_hash[i] << (8 * i)));
                    }

                    uint patternId = (uint)(hash % Owner.SPARSE_SIZE);
                    
                    Owner.Output.Host[patternId] = 1.0f;
                }


                Owner.Output.SafeCopyToDevice();
            }
        }

        public override void UpdateMemoryBlocks()
        {
            if (HashMethod == MyHashMethod.MD5Dense)
            {
                OutputSize = 128;
            }
            else if (HashMethod == MyHashMethod.MD5Sparse)
            {
                OutputSize = SPARSE_SIZE;
            }
        }
    }
}
