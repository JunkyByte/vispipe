import React, {useState, useEffect} from 'react';
import { makeStyles } from '@material-ui/core/styles';
import ListSubheader from '@material-ui/core/ListSubheader';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';
import Collapse from '@material-ui/core/Collapse';
import ExpandLess from '@material-ui/icons/ExpandLess';
import ExpandMore from '@material-ui/icons/ExpandMore';
import * as APP from "./App"

const open = []

const List_style = makeStyles(theme => ({
    '@global': {
        '*::-webkit-scrollbar': {
            display: 'none'
        },
    },
    root: {
      width: 200,
      maxWidth: 360,
    //   backgroundColor: theme.palette.background.paper,
    },
    nested: {
      paddingLeft: theme.spacing(4),
    },
  }));

  const Item_style = makeStyles({
    root: {
        zIndex: 10
    },
})

function SidebarItem ({label, items, depthStep = 10, depth = 0, onClick, ...rest}){
    const [block, setBlock] = useState({})
    const classes = Item_style()
    const [open, setOpen] = useState(false)

    useEffect(() => {
        if (typeof items != 'object'){
            fetch("/get_block", {
                method : "POST",
                body: JSON.stringify({"name" : label}),
                headers: new Headers({
                    "content-type" : "application/json"
                })
            }).then(res => res.json()).then(data => setBlock(data))
        }
        
    }, [])



    return (
        <>
            <ListItem button dense {...rest} className={classes.root} onClick={() => Array.isArray(items) ? (onClick(), setOpen(!open)) : APP.pipeline.spawn_node(block)}>
                <ListItemText style={{ paddingLeft: depth * depthStep}}>
                    <span>{label}</span>
                </ListItemText>
                { Array.isArray(items) ? (open ? <ExpandLess/> : <ExpandMore/>) : null}
            </ListItem>
            <Collapse in={open} timeout="auto" unmountOnExit>
                {Array.isArray(items) ? (
                    <List disablePadding dense>
                        {items.map((subitem) => (
                            <SidebarItem 
                                key={subitem.name}
                                depth={depth + 1}
                                depthStep = {depthStep}
                                {...subitem}
                            />
                        ))}
                    </List>
                ) : null}
            </Collapse>
        </>
    )
}

  export default function NewSideBar({depthStep, depth}){
    const classes = List_style()

    const [menu, setMenu] = React.useState([])

    React.useEffect(()=> {
        fetch("/get_menu").then(res => res.json()).then(data => {
            const menu_data = Object.values(data)
            setMenu(menu_data)
        })
    }, [])
      

    for(var i = 0; i<menu.length; i++){
        open.push(false)
    }

    const handleClick = (index) => {
        open[index] = !open[index]
        console.log(open)
    }

    return (
        <div id="sidebar">
        <List
            component="nav"
            aria-labelledby="nested-list-subheader"
            className={classes.root}
        >
            {menu.map((sidebarItem, index) => (
            <SidebarItem
                onClick={() => handleClick(index)}
                key={`${sidebarItem.name}${index}`}
                depthStep={depthStep}
                depth={depth}
                {...sidebarItem}
            />
            ))}
        </List>
    </div>
    )
  }
