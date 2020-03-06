import React, {useState, useEffect} from 'react'


function Sidebar(){
    const [menu, setMenu] = useState(0);

    useEffect(()=> {
        fetch('/get_menu').then(res => res.json()).then(data => {
            const menu_data = Object.values(data)
            console.log(menu_data)
        })
    }, [])

    return(
        <div></div>
    )
}

export default Sidebar