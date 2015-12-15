## Loop group example

This sample project shows usage of `LoopGroup` and `ConditionalGate`.

### Loop group
`LoopGroup` node is a node group which run its content multiple times in one simulation step. You set the number of iterations in node's `Iterations` parameter.

You can also set node's `LoopType` parameter to one of following values:

- `Normal` - group loops only those tasks, which have `OneShot` set to `False` (which is default)
- `ALL` - group loops all the tasks inside

### Conditional gate
Conditional gate is designed for usage together with `LoopGroup`. It takes two inputs and sends them to its output according to its `Gate Inputs` task `Iteration Threshold` parameter.

`ConditionalGate` internally knows which `LoopGroup` iteration is running right now and it outputs the iteration number to its `Iteration` output.

If the actual iteration number is lower or equal to `Iteration Threshold` parameter, `InputA` is copied to `Output`. Otherwise `InputB` is copied to `Output`. Iteration number is resetted each simulation step.

### Example usage
Brain: [BrainSimulatorSampleProjects\LoopGroup\loopgroupexample.brain](https://github.com/GoodAI/BrainSimulatorSampleProjects/blob/master/LoopGroup/loopGroupExample.brain)

Let's say you want have an image and want to blur it. But `Blur` which is provided by `Filter2D` node is not strong enough for you and you decide to blur image multiple times. You can connect multiple `Filter2D` nodes in series or you can use `LoopGroup` and `ConditionalGate` as in following image

![Internals of loop group](../img/loopgroupexample.png)