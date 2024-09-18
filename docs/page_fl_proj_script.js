"use strict"; 

document.addEventListener("DOMContentLoaded", () => {
    // DOM elements
    const apiTabLink = document.getElementById("api-doc-tab-link");
    const mainTabLink = document.getElementById("main-tab-link"); 
    const bodyEl = document.querySelector("body");
    const footer = document.getElementById("page-footer");

    // Set the Main tab focused (selected) if API Doc opened on the additional tab
    apiTabLink.addEventListener("click", ()=>{
        mainTabLink.focus();  // remain focus on the "Main" tab, because other tab opens other tab
    });

    // Set the height of body element equal to vh measurements if it more than 100 % of height css value
    // which is calculated depending on the content size
    if (window.innerHeight > bodyEl.clientHeight) {
        bodyEl.style.height = "99.6vh";
    }

    // Provide programmatically changing of the class names of Bootstrap acquired via CDN for more adaptivity on page loading
    // Note that the redefining Bootstrap classes is possible via SASS
    if (window.innerWidth< 1201) {
        const classesList = footer.classList;  // get all classes defined for using of Bootstrap
        classesList.remove("justify-content-start"); classesList.add("justify-content-center");
        classesList.remove("ms-5"); classesList.remove("me-1"); classesList.add("mx-2");
    }
});
