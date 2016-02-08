using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;

namespace GoodAI.Core.Memory
{
    public sealed class MyMemoryManager
    {
        private static MyMemoryManager SINGLETON;
        
        public static MyMemoryManager Instance
        {
            get
            {
                //return MyKernelFactoryOld.Instance;
                if (SINGLETON == null)
                {
                    SINGLETON = new MyMemoryManager();
                }
                return SINGLETON;
            }
        }

        /// <summary>
        /// A disposable backup of a MyMemoryManager instance.
        /// </summary>
        public class Backup : IDisposable
        {
            private MyMemoryManager m_instance;

            public Backup(MyMemoryManager instance)
            {
                m_instance = instance;
            }

            /// <summary>
            /// Forget any backup so that Dispose doesn't use it.
            /// </summary>
            public void Forget()
            {
                m_instance = null;
            }

            /// <summary>
            /// Restore the backup unless it's been explicitely forgotten.
            /// </summary>
            public void Dispose()
            {
                if (m_instance != null)
                    SINGLETON = m_instance;  // Restore backup
            }
        }

        /// <summary>
        /// Get a backup of the memory manager.
        /// </summary>
        /// <returns>A disposable backup that auto-restores when not forgotten.</returns>
        public static Backup GetBackup()
        {
            var backup = new Backup(SINGLETON);

            // Force creating a new instance when Instance is called.
            SINGLETON = null;

            return backup;
        }

        //TODO: Should be something much sofisticated later, like virtual memory
        private Dictionary<MyNode, List<MyAbstractMemoryBlock>> m_memoryBlocks;
        private Dictionary<string, IDisposable>[] m_globalVariables;
        private Dictionary<string, CUdeviceptr>[] m_globalVarPointers;

        private MyMemoryManager() 
        {
            m_memoryBlocks = new Dictionary<MyNode, List<MyAbstractMemoryBlock>>();
            m_globalVariables = new Dictionary<string, IDisposable>[MyKernelFactory.Instance.DevCount];
            m_globalVarPointers = new Dictionary<string, CUdeviceptr>[MyKernelFactory.Instance.DevCount];

            for (int i = 0; i < m_globalVariables.Length; i++)
            {
                m_globalVariables[i] = new Dictionary<string, IDisposable>();
                m_globalVarPointers[i] = new Dictionary<string, CUdeviceptr>();
            }
        }

        public MyMemoryBlock<T> CreateMemoryBlock<T>(MyNode holder) where T : struct
        {
            return (MyMemoryBlock<T>)CreateMemoryBlock(holder, typeof(MyMemoryBlock<T>));
        }

        internal MyAbstractMemoryBlock CreateMemoryBlock(MyNode holder, Type blockType)
        {
            MyAbstractMemoryBlock block = (MyAbstractMemoryBlock)Activator.CreateInstance(blockType, true);
            block.Owner = holder;

            if (!m_memoryBlocks.ContainsKey(holder))
            {
                m_memoryBlocks[holder] = new List<MyAbstractMemoryBlock>();
            }
            m_memoryBlocks[holder].Add(block);

            return block;
        }

        private static readonly List<MyAbstractMemoryBlock> EMPTY_LIST = new List<MyAbstractMemoryBlock>();

        public List<MyAbstractMemoryBlock> GetBlocks(MyNode holder)
        {
            if (m_memoryBlocks.ContainsKey(holder))
            {
                return m_memoryBlocks[holder];
            }
            else return EMPTY_LIST;
        }

        public bool IsRegistered(MyNode holder)
        {
            return m_memoryBlocks.ContainsKey(holder);
        }

        public MyAbstractMemoryBlock GetMemoryBlockByName(MyNode holder, string name)
        {
            List<MyAbstractMemoryBlock> blocks = GetBlocks(holder);

            foreach (MyAbstractMemoryBlock block in blocks)
            {
                if (block.Name == name)
                {
                    return block;
                }
            }
            return null;
        }

        public delegate T[] GlobalVariableInitializer<T>();

        public CudaDeviceVariable<T> GetGlobalVariable<T>(string name, int nGPU, GlobalVariableInitializer<T> initializer) where T : struct
        {
            if (!m_globalVariables[nGPU].ContainsKey(name)) {

                T[] initValues = initializer();
                
                CudaDeviceVariable<T> variable = new CudaDeviceVariable<T>(
                    MyKernelFactory.Instance.GetContextByGPU(nGPU).AllocateMemory(
                    initValues.Length * System.Runtime.InteropServices.Marshal.SizeOf(typeof(T))));

                variable.CopyToDevice(initValues);
                m_globalVariables[nGPU][name] = variable;
                m_globalVarPointers[nGPU][name] = variable.DevicePointer;
            }

            return (CudaDeviceVariable<T>) m_globalVariables[nGPU][name];
        }

        public void ClearGlobalVariable(string name, int nGPU)
        {
            if (m_globalVariables[nGPU].ContainsKey(name))
            {
                MyKernelFactory.Instance.GetContextByGPU(nGPU).FreeMemory(m_globalVarPointers[nGPU][name]);     
                m_globalVariables[nGPU][name].Dispose();

                m_globalVariables[nGPU].Remove(name);
                m_globalVarPointers[nGPU].Remove(name);
            }
        }

        public void ClearGlobalVariables()
        {
            for (int i = 0; i < m_globalVariables.Length; i++)
            {
                foreach (CUdeviceptr pointer in m_globalVarPointers[i].Values)
                {
                    MyKernelFactory.Instance.GetContextByGPU(i).FreeMemory(pointer);     
                }

                foreach (IDisposable variable in m_globalVariables[i].Values)
                {                    
                    variable.Dispose();
                }

                m_globalVariables[i].Clear();
                m_globalVarPointers[i].Clear();
            }            
        }

        private delegate void MemoryAction(MyAbstractMemoryBlock block);

        private void IterateBlocks(MyNode holder, bool recursive, MemoryAction action)
        {
            List<MyAbstractMemoryBlock> myListOfBlocks = GetBlocks(holder);

            for (int i = 0; i < myListOfBlocks.Count; i++)
            {                
                action(myListOfBlocks[i]);
            }

            if (holder is MyNodeGroup)
            {
                for (int i = 0; i < (holder as MyNodeGroup).Children.Count; i++)                
                {
                    IterateBlocks((holder as MyNodeGroup).Children[i], recursive, action);
                }
            }
        }

        public void AllocateBlocks(MyNode holder, bool recursive, bool host = true, bool device = true)
        {
            MemoryAction action = delegate(MyAbstractMemoryBlock memoryBlock)
            {
                if (memoryBlock.Owner.GPU < 0)
                {
                    MyLog.WARNING.WriteLine("No GPU assigned for node " + memoryBlock.Owner.Name + ". Using default GPU (" + (MyKernelFactory.Instance.DevCount - 1) + ").");
                    memoryBlock.Owner.GPU = MyKernelFactory.Instance.DevCount - 1;
                }

                if (host) memoryBlock.AllocateHost();
                if (device) memoryBlock.AllocateDevice();
            };

            IterateBlocks(holder, recursive, action);
        }

        public void FreeBlocks(MyNode holder, bool recursive, bool host = true, bool device = true)
        {
            MemoryAction action = delegate(MyAbstractMemoryBlock memoryBlock)
            {
                if (host) memoryBlock.FreeHost();
                if (device) memoryBlock.FreeDevice();
            };

            IterateBlocks(holder, recursive, action);
        }

        public SizeT SizeOf(MyNode holder, bool recursive)
        {
            SizeT totalMemory = 0;

            MemoryAction action = delegate(MyAbstractMemoryBlock memoryBlock)
            {
                totalMemory += memoryBlock.GetSize();
            };

            IterateBlocks(holder, recursive, action);

            return totalMemory;
        }

        public void RemoveBlocks(MyNode holder)
        {            
            m_memoryBlocks.Remove(holder);
        }

        public void RemoveBlock(MyNode holder, MyAbstractMemoryBlock block)
        {
            if (m_memoryBlocks.ContainsKey(holder)) {
                m_memoryBlocks[holder].Remove(block);

                if (m_memoryBlocks[holder].Count == 0)
                {
                    m_memoryBlocks.Remove(holder);
                }
            }
        }

        public void SynchronizeSharedBlocks(MyNode holder, bool recursive)
        {
            MemoryAction action = delegate(MyAbstractMemoryBlock memoryBlock)
            {
                memoryBlock.Synchronize();
            };

            IterateBlocks(holder, recursive, action);
        }

        public void SaveBlocks(MyNode holder, bool recursive, string path) 
        {
            MyMemoryBlockSerializer serializer = new MyMemoryBlockSerializer();

            MemoryAction action = delegate(MyAbstractMemoryBlock memoryBlock)
            {
                if (memoryBlock.Persistable)
                {
                    serializer.SaveBlock(memoryBlock);
                }
            };

            IterateBlocks(holder, true, action);
        }

        public void LoadBlocks(MyNode holder, bool recursive, string path)
        {
            MyMemoryBlockSerializer serializer = new MyMemoryBlockSerializer();

            MemoryAction action = delegate(MyAbstractMemoryBlock memoryBlock)
            {
                if (memoryBlock.Persistable)
                {
                    serializer.LoadBlock(memoryBlock, path);                              
                }
            };

            IterateBlocks(holder, true, action);
        }

        // TODO: remove?
        private IDictionary<string, MyAbstractMemoryBlock> CollectMemoryBlocks()
        {
            var memBlocks = new Dictionary<string, MyAbstractMemoryBlock>();

            foreach (var memoryBlockList in m_memoryBlocks.Values)
            {
                foreach (MyAbstractMemoryBlock memoryBlock in memoryBlockList)
                {
                    string memBlockName = MyMemoryBlockSerializer.GetUniqueName(memoryBlock);

                    if (!memBlocks.ContainsKey(memBlockName))
                        memBlocks.Add(memBlockName, memoryBlock);
                }
            }

            return memBlocks;
        }
    }
}
