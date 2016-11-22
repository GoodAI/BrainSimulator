import os
import clr
import types
from _winreg import OpenKey, QueryValueEx, HKEY_LOCAL_MACHINE


def _configureContainer():
    conf = CoreContainerConfiguration()
    conf.Configure(TypeMap.SimpleInjectorContainer)


def _configureModulesAndKernels(brainsim_path):
    module_dir = os.path.join(brainsim_path, 'modules')
    for module_name in os.listdir(module_dir):
        module_file = os.path.join(module_dir, module_name, module_name + '.dll')
        MyConfiguration.ModulesSearchPath.Add(module_file)
        if os.path.isfile(module_file):
            clr.AddReference(module_file)
        else:
            print("Unable to find file {}. Classes from this module will not be available in Python code.".format(absPath))
    MyConfiguration.GlobalPTXFolder = "{}/GoodAI.BasicNodes/ptx/".format(module_dir)


def _init(brainsim_path):
    _configureContainer()
    _configureModulesAndKernels(brainsim_path)


def _load_classes():
    global MyProjectRunner, MyConfiguration, CoreContainerConfiguration, TypeMap

    from GoodAI.Core.Execution import MyProjectRunner as DefaultProjectRunner
    from GoodAI.Core.Configuration import MyConfiguration
    from GoodAI.Platform.Core.Configuration import CoreContainerConfiguration
    from GoodAI.TypeMapping import TypeMap

    class MyProjectRunner(DefaultProjectRunner):

        # you cannot use number as attrs - therefore this
        def __getitem__(self, idx):
            node = self.GetNode(idx)
            return node


def _brainsim_path_from_registry():
    key = OpenKey(HKEY_LOCAL_MACHINE, "SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\BrainSimulator.exe")
    val = QueryValueEx(key, "Path")
    return val[0]


def load_brainsim(brainsim_path=None):
    if not brainsim_path:
        try:
            brainsim_path = _brainsim_path_from_registry()
            print("Using BrainSimulator path from registry: {}".format(brainsim_path))
        except WindowsError:
            pass
    else:
        print("Using user-specified path to BrainSimulator: {}".format(brainsim_path))
    if not brainsim_path:
        brainsim_path = "C:\Program Files\GoodAI\Brain Simulator"
        print("Using default BrainSimulator path: {}".format(brainsim_path))

    if not os.path.isdir(brainsim_path):
        print("Path {} does not exist. Unable to load Brain Simulator.".format(brainsim_path))
        return False

    absPath = os.path.join(brainsim_path, 'GoodAI.Platform.Core.dll')
    if not os.path.isfile(absPath):
        print("Unable to find file {}. Did you specify the path to the latest Brain Simulator installation?".format(absPath))
        return False
    clr.AddReference(absPath)

    _load_classes()
    _init(brainsim_path)
    return True
