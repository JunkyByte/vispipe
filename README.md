# VisPipe
## Build Pipelines visually and run them via code
[![CodeFactor](https://www.codefactor.io/repository/github/junkybyte/vispipe/badge/master?s=b4f0ed72fedffa8ed8cbc9bc9887a0db528a24b2)](https://www.codefactor.io/repository/github/junkybyte/vispipe/overview/master)
![Package Testing](https://github.com/JunkyByte/vispipe/workflows/Package%20Testing/badge.svg?branch=master)

VisPipe is a python package that can help you build pipelines by providing a convenient visual creation tool that can help you debug and identify problems before they happen.
Once you are satisfied with the result a pipeline can be saved to file and run via code, the outputs of it can be easily interacted with through pure python code.

By default VisPipe provides a number of Operational blocks, you are encouraged to extend them by creating your own.
VisPipe will run using python `Threads` or `Process` (+ `Queues`) internally to reduce the tradeoff between performance and flexibility.
Each block of your pipeline will support multiple input/output arguments, multiple connections and custom static arguments.

## Installation
WIP
