# goodai-brainsim
Python module for using GoodAI's Brain Simulator

The module itself serves as an interface to Brain Simulator's MyProjectRunner. You can find a documentation to MyProjectRunner on [GoodAI's docs website](http://docs.goodai.com/brainsimulator/guides/projectrunner/index.html).

# Install from source

Go to folder with `setup.py`

Run `pip install .`

Note: `GoodAI-BrainSim` module depends on [Python for .NET](http://pythonnet.github.io/) which cannot be installed using easy_install, therefore you cannot use the usual `python setup.py install`


# How to use module

- First thing you need to do is to import a `load_brainsim` method from the module
- Then you have to call this method. It has one optional argument in which you can specify a path to the folder where you have Brain Simulator installed. If you do not provide the path, it will try to find a path itself in the Windows registry. If the method is unable to find a path in registry, it will use the path "C:\Program Files\GoodAI\Brain Simulator".\
- After that, you can import other types from the `goodai.brainsim` module
    + You will always want to import `MyProjectRunner`. It is used to control the Brain Simulator
    + If you want to use Brain Simulator's School for AI, you want to import `SchoolCurriculum`, `CurriculumManager` and `LearningTaskFactory` as well
    + You can also import multiple other types useful e.g. for curriculum creation. Those are `ToyWorldAdapterWorld`

Together, beginning of your script may look like

``` python
from goodai.brainsim import load_brainsim
load_brainsim("C:\Users\john.doe\my\brainsimulator\path")

from goodai.brainsim import MyProjectRunner, SchoolCurriculum, CurriculumManager, ToyWorldAdapterWorld, LearningTaskFactory
```