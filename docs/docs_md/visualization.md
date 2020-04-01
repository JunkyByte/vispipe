# Visualization

The visualization can be useful to build pipelines in an easier way, to have your custom
blocks inside the visualization you should import the `.py` files and launch the server
from there:
```python
from vispipe import Server
import my_custom_blocks
import my_other_custom_blocks

if __name__ == '__main__':
    # slow=True is recommended and will run the pipeline slowly to allow visualization
    # path is the checkpoint path you want to use
    Server(path, slow=True)
```

Once the server is launched connect to `localhost:5000`.

### Add Nodes
The nodes have a tag that can be specified during declaration, you will find these tags during visualization.
```python
@block(tag='my_custom_tag')
def f():
    # (...)
```
To spawn a node simply click it on the right side menu.
You can create switch between tags using the right top side arrows.

### Add Connections
![Missing gif](https://media.giphy.com/media/idSuAhb6Wa6rRvzIkT/giphy.gif)

### Set custom arguments
![Missing gif](https://media.giphy.com/media/USEG1wMYmUL12AUD2C/giphy.gif)

### Add outputs
![Missing gif](https://media.giphy.com/media/VdQsT8i1gj2CUMH07D/giphy.gif)

### Add visualization
<img src="https://media.giphy.com/media/RGdYD6vdMbNVQP0sUO/giphy.gif" width="480"/>
