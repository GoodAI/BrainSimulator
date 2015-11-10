using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Graph.Compatibility {
	/// <summary>
	/// Describes a strategy to compare two node item connectors for a possible connection.
	/// </summary>
	public interface ICompatibilityStrategy {
		/// <summary>
		/// Determine if two node item connectors could be connected to each other.
		/// </summary>
		/// <param name="from">From which node connector are we connecting.</param>
		/// <param name="to">To which node connector are we connecting?</param>
		/// <returns><see langword="true"/> if the connection is valid; <see langword="false"/> otherwise</returns>
		bool CanConnect(NodeConnector from, NodeConnector to);
	}
}
