using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using GoodAI.BrainSimulator.Utils;
using Xunit;

namespace CoreTests
{
    public class UndoManagerTests
    {
        [Fact]
        public void Undoes()
        {
            var steps = new[] { "a", "b" }.Select(s => new ProjectState(s)).ToList();

            var manager = new UndoManager(5);

            foreach (ProjectState step in steps)
                manager.SaveState(step);

            manager.SaveState(new ProjectState("c"));

            var result = new List<ProjectState>();
            for (int i = 0; i < 2; i++)
                result.Add(manager.Undo());

            result.Reverse();

            Assert.Equal(steps, result);
        }

        [Fact]
        public void Redoes()
        {
            var steps = new[] { "b", "c" }.Select(s => new ProjectState(s)).ToList();

            var manager = new UndoManager(5);

            manager.SaveState(new ProjectState("a"));

            foreach (ProjectState step in steps)
                manager.SaveState(step);

            for (int i = 0; i < 2; i++)
                manager.Undo();

            var result = new List<ProjectState>();
            for (int i = 0; i < 2; i++)
                result.Add(manager.Redo());

            Assert.Equal(steps, result);
        }

        [Fact]
        public void ReturnsNullWhenOutOfItems()
        {
            var manager = new UndoManager(3);
            manager.SaveState(new ProjectState(""));

            manager.Undo();
            Assert.Null(manager.Undo());

            manager.Redo();
            Assert.Null(manager.Redo());
        }

        [Fact]
        public void LimitsHistoryCount()
        {
            const uint size = 5u;
            var manager = new UndoManager(size);

            for (int i = 0; i < size * 2; i++)
                manager.SaveState(new ProjectState(""));

            uint counter = 0;
            while (manager.Undo() != null)
                counter++;

            Assert.Equal(size, counter);
        }

        [Fact]
        public void WorksWithZeroCapacity()
        {
            var manager = new UndoManager(0);
            manager.SaveState(new ProjectState(""));
            Assert.Null(manager.Undo());
            Assert.Null(manager.Redo());
        }

        [Fact]
        public void IndicatesUndoPossible()
        {
            var manager = new UndoManager(5);
            manager.SaveState(new ProjectState(""));
            Assert.False(manager.CanUndo());

            manager.SaveState(new ProjectState(""));
            Assert.True(manager.CanUndo());

            manager.Undo();
            Assert.False(manager.CanUndo());
        }

        [Fact]
        public void IndicatesRedoPossible()
        {
            var manager = new UndoManager(5);
            manager.SaveState(new ProjectState(""));
            Assert.False(manager.CanRedo());

            manager.SaveState(new ProjectState(""));
            Assert.False(manager.CanRedo());
            manager.Undo();

            Assert.True(manager.CanRedo());
        }
    }
}
