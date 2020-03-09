import React from "react"
import { makeStyles } from "@material-ui/core/styles"
import BottomNavigation from '@material-ui/core/BottomNavigation';
import BottomNavigationAction from '@material-ui/core/BottomNavigationAction';
import PlayArrowIcon from '@material-ui/icons/PlayArrow';
import StopIcon from '@material-ui/icons/Stop';
import ClearIcon from '@material-ui/icons/Clear';
import * as APP from "./App"



const useStyles = makeStyles({
    root : {
        width : '100%',
        position: "fixed",
        bottom: 0,
        zIndex: -10
    },
})

export default function BottomMenu() {
    const classes = useStyles();
    const [value, setValue] = React.useState(0)

    return (
        <div id="nav-bar">
            <BottomNavigation value={value} onChange={(event, newValue) => setValue(newValue)} showLabels className={classes.root}>
                <BottomNavigationAction label="Run" icon={<PlayArrowIcon/>} onClick={()=> APP.pipeline.run_pipeline()}/>
                <BottomNavigationAction label="Stop" icon={<StopIcon/>} onClick={() => APP.pipeline.stop_pipeline()}/>
                <BottomNavigationAction label="Clear" icon={<ClearIcon/>} onClick={()=> APP.pipeline.clear_pipeline()}/>
            </BottomNavigation>
        </div>
    )
}