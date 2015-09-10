
namespace GoodAI.Core.Task
{
    public interface IMyExecutable
    {
        void Execute();
        bool Enabled { get; }        
        uint SimulationStep { get; set; }
        string Name { get; }
    }
}
