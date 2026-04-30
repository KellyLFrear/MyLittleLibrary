
// Turns Pages When Clicking "Next" Or "Previous" Buttons
const pageTurnBtn = document.querySelectorAll('.nextprev-btn');

pageTurnBtn.forEach((el, index) => {
    el.onclick = () => {
        const pageTurnId = el.getAttribute('data-page');
        const pageTurn = document.getElementById(pageTurnId);

        // Turns The Page Back (Right -> Left)
        if (pageTurn.classList.contains('turn')) {
            pageTurn.classList.remove('turn');
            setTimeout(() => {
                pageTurn.style.zIndex = 20 - index;
            }, 500)
        }
        // Turns The Page Forward (Left -> Right)
        else {
            pageTurn.classList.add('turn');
            setTimeout(() => {
                pageTurn.style.zIndex = 20 + index;
            }, 500)
        }
    }
})


// "Generate Story" Button When Clicked
const pages = document.querySelectorAll('.book-page.page-right');
const generateStoryPageBtn = document.querySelector('.btn.generate-story-page-btn');

generateStoryPageBtn.onclick = () => {
    pages.forEach((pageTurnBtn, index) => {
        setTimeout(() => {
            pageTurnBtn.classList.add('turn');

            setTimeout(() => {
                pageTurnBtn.style.zIndex = 20 + index;
            }, 500)

        }, (index +1) * 200 + 100)
    })
}


// Library Button (On "Profile Page") When Clicked
const addToLibraryBtn = document.querySelector('.btn.add-to-library-page-btn');

addToLibraryBtn.onclick = () => {
    pages.forEach((page, index) => {

        // Only Flip Up To Page 2 (Index 0 And 1)
        if (index < 2) {
            setTimeout(() => {
                page.classList.add('turn');

                setTimeout(() => {
                    page.style.zIndex = 20 + index;
                }, 500);

            }, (index + 1) * 200 + 100);
        }

    });
};


// Profile Button (On "Generate Story" Page) When Clicked
const backProfileBtn = document.querySelector('.back-profile');

backProfileBtn.onclick = () => {
    pages.forEach((_, index) => {
        setTimeout(() => {
            const currentPage = pages[pages.length - 1 - index];

            currentPage.classList.remove('turn');

            setTimeout(() => {
                currentPage.style.zIndex = 10 + index;
            }, 500);

        }, (index + 1) * 200 + 100);
    });
};


// Opening Animation
const coverRight = document.querySelector('.cover.cover-right');
const pageLeft = document.querySelector('.book-page.page-left');

// Opening Animation (Cover Right Animation)
setTimeout(() => {
    coverRight.classList.add('turn');
}, 2100)

setTimeout(() => {
    coverRight.style.zIndex = -1;
}, 2800)

// Opening Animation (Page Left or Profile Page Animation)
setTimeout(() => {
    pageLeft.style.zIndex = -1;
}, 3200)

 // Opening Animation (All Page Right Animation)
 pages.forEach((_, index) => {
    setTimeout(() => {
        const currentPage = pages[pages.length - 1 - index];

        currentPage.classList.remove('turn');

        setTimeout(() => {
            currentPage.style.zIndex = 10 + index;
        }, 500);

    }, (index + 1) * 200 + 2100);
});