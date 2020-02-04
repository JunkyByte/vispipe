class AbstractNode {
    constructor(block) {
        this.block = block;
    }
}

class StaticNode extends AbstractNode {
    constructor(block) {
        super(block);
    }
}

class Node extends AbstractNode {
    constructor(block) {
        super(block);
    }

}

class Block {
    constructor(name, input_args, custom_args, output_names) {
        this.name = name;
        this.input_args = input_args;
        this.custom_args = custom_args;
        this.output_names = output_names;
    }
}