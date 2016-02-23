using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

// General Information about an assembly is controlled through the following 
// set of attributes. Change these attribute values to modify the information
// associated with an assembly.
[assembly: AssemblyTitle("GoodAI.Platform.Core")]
[assembly: AssemblyDescription("Core Library for simulation platform")]
[assembly: AssemblyConfiguration("")]
[assembly: AssemblyCompany("GoodAI")]
[assembly: AssemblyProduct("Brain Simulator")]
[assembly: AssemblyCopyright("Copyright © 2014-2016")]
[assembly: AssemblyTrademark("")]
[assembly: AssemblyCulture("")]

// Setting ComVisible to false makes the types in this assembly not visible 
// to COM components.  If you need to access a type in this assembly from 
// COM, set the ComVisible attribute to true on that type.
[assembly: ComVisible(false)]

// The following GUID is for the ID of the typelib if this project is exposed to COM
[assembly: Guid("7dc97e6d-d8ef-4f86-be83-dae96f7727c0")]

// Version information for an assembly consists of the following four values:
//
//      Major Version
//      Minor Version 
//      Build Number
//      Revision
//
// You can specify all the values or you can default the Build and Revision Numbers 
// by using the '*' as shown below:
// [assembly: AssemblyVersion("1.0.*")]
[assembly: AssemblyVersion("0.5.0.0")]
[assembly: AssemblyFileVersion("0.5.0.0")]

// Testing
[assembly: InternalsVisibleTo("BrainSimulatorTests")]
[assembly: InternalsVisibleTo("CoreTests")]  // e.g. for MyForkTests

