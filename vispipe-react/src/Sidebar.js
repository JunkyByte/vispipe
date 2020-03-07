import React, {useState, useEffect} from 'react'
import List from "@material-ui/core/List"
import ListItem from "@material-ui/core/ListItem"
import ListItemText from "@material-ui/core/ListItemText"
import {withStyles} from "@material-ui/core/styles"
import * as APP from "./App"

const StyledListItem = withStyles({
    root: {
      background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)',
      borderRadius: 3,
      border: 0,
      color: 'white',
      height: 48,
      padding: '0 30px',
      boxShadow: '0 3px 5px 2px rgba(255, 105, 135, .3)',
    },
    label: {
      textTransform: 'capitalize',
    },
  })(ListItem);

function SidebarItem ({label, items, depthStep = 10, depth = 0, ...rest}){
    const [block, setBlock] = useState({})
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
            <ListItem button dense onClick={()=>APP.pipeline.spawn_node(block)} {...rest}>
                <ListItemText style={{ paddingLeft: depth * depthStep}}>
                    <span>{label}</span>
                </ListItemText>
            </ListItem>
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
        </>
    )
}


function Sidebar({items, depthStep, depth}){
    const [menu, setMenu] = useState([]);

    useEffect(()=> {
        fetch('/get_menu').then(res => res.json()).then(data => {
            const menu_data = Object.values(data)
            setMenu(menu_data)
        })
    }, [])

    return(
        <div className="sidebar">
            <List disablePadding dense>
                {menu.map((sidebarItem, index) => (
                <SidebarItem
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

export default Sidebar