import React from "react"
import { makeStyles } from "@material-ui/core/styles"
import BottomNavigation from '@material-ui/core/BottomNavigation';
import BottomNavigationAction from '@material-ui/core/BottomNavigationAction';
import PlayArrowIcon from '@material-ui/icons/PlayArrow';
import StopIcon from '@material-ui/icons/Stop';



const useStyles = makeStyles({
    root : {
        width : 500,
    },
})

export default function BottomMenu() {
    const classes = useStyles();
    const [value, setValue] = React.useState(0)

    return (
        <div id="nav-bar">
            <BottomNavigation value={value} onChange={(event, newValue) => setValue(newValue)} showLabels className={classes.root}>
                <BottomNavigationAction label="Run" icon={<PlayArrowIcon/>}/>
                <BottomNavigationAction label="Stop" icon={<StopIcon/>}/>
            </BottomNavigation>
        </div>
    )
}