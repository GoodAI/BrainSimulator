using System.Windows;

namespace Touchless.Vision.Contracts
{
    public interface ITouchlessAddIn
    {
        string Name { get; }
        string Description { get; }
        bool HasConfiguration { get; }
        UIElement ConfigurationElement { get; }
    }
}