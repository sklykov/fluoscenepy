"use strict"; 

document.addEventListener("DOMContentLoaded", () => {
    
    // DOM elements
    const apiTabLink = document.getElementById("api-doc-tab-link");
    const mainTabLink = document.getElementById("main-tab-link"); 
    const bodyEl = document.querySelector("body");

    // Set the Main tab focused if API Doc opened
    apiTabLink.addEventListener("click", ()=>{
        mainTabLink.focus();  // remain focus on the "Main" tab, because other tab opens other tab
    });

    // Set the height of body element equal to vh measurements if it more than 100 % of height css value
    // which is calculated depending on the content size
    if (window.innerHeight > bodyEl.clientHeight) {
        bodyEl.style.height = "99.6vh";
    }
});
