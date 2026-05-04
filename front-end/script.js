
// ════════════════════════════════════════════════════════════════════════════
// Page-turn animations (original logic — unchanged)
// ════════════════════════════════════════════════════════════════════════════

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


// ════════════════════════════════════════════════════════════════════════════
// Backend API integration
// ════════════════════════════════════════════════════════════════════════════

// ── Helpers ──────────────────────────────────────────────────────────────────

async function apiFetch(url, options = {}) {
    const res = await fetch(url, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
    });
    if (res.status === 401) {
        // Session expired — redirect to login
        window.location.href = '/';
        return null;
    }
    return res;
}

function buildBookCard(title, level, detail) {
    const div = document.createElement('div');
    div.className = 'bookbox-content';
    div.innerHTML = `
        <h3>${escapeHtml(title)}</h3>
        ${level ? `<span class="reading_level"><i class="bx bx-book"></i>Reading Level: ${escapeHtml(level)}</span>` : ''}
        ${detail ? `<p>${escapeHtml(detail)}</p>` : ''}
    `;
    return div;
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// ── Load profile on page open ─────────────────────────────────────────────────

async function loadProfile() {
    const res = await apiFetch('/api/profile');
    if (!res) return;
    if (!res.ok) return;

    const data = await res.json();

    const nameEl  = document.getElementById('profile-name');
    const levelEl = document.getElementById('profile-level');

    if (nameEl) {
        nameEl.textContent = data.display_name || data.username || 'User';
    }
    if (levelEl) {
        const lvl = (data.current_profile && data.current_profile.estimated_level)
            || data.avg_reading_lvl
            || '—';
        levelEl.textContent = `Average Reading Level : ${lvl}`;
    }
}

// ── Load book recommendations ─────────────────────────────────────────────────

async function loadRecommendations() {
    const container = document.getElementById('suggestions-list');
    if (!container) return;

    container.innerHTML = '<p style="padding:8px;">Loading suggestions…</p>';

    const res = await apiFetch('/api/recommendations');
    if (!res || !res.ok) {
        container.innerHTML = '<p style="padding:8px;">Could not load suggestions.</p>';
        return;
    }

    const { recommendations } = await res.json();
    container.innerHTML = '';

    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = '<p style="padding:8px;">No suggestions yet. Type a topic below and click Generate!</p>';
        return;
    }

    recommendations.forEach(rec => {
        const card = buildBookCard(
            rec.title || 'Unknown Title',
            rec.recommended_level,
            rec.reason,
        );
        container.appendChild(card);
    });
}

// ── Book suggestion form (generate new recommendations) ────────────────────────

const suggestionsForm = document.getElementById('suggestions-form');
if (suggestionsForm) {
    suggestionsForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = document.getElementById('suggestions-query').value.trim();
        const container = document.getElementById('suggestions-list');
        container.innerHTML = '<p style="padding:8px;">Generating suggestions…</p>';

        const res = await apiFetch('/api/recommendations/generate', {
            method: 'POST',
            body: JSON.stringify({ query }),
        });
        if (!res || !res.ok) {
            container.innerHTML = '<p style="padding:8px;">Could not generate suggestions.</p>';
            return;
        }

        const { recommendations, note } = await res.json();
        container.innerHTML = '';

        if (note) {
            const p = document.createElement('p');
            p.style.cssText = 'padding:4px;font-size:0.8em;color:#aaa;';
            p.textContent = note;
            container.appendChild(p);
        }

        if (!recommendations || recommendations.length === 0) {
            container.innerHTML += '<p style="padding:8px;">No suggestions found.</p>';
            return;
        }

        recommendations.forEach(rec => {
            const card = buildBookCard(
                rec.title || 'Unknown Title',
                rec.recommended_level,
                rec.reason,
            );
            container.appendChild(card);
        });
    });
}

// ── Load user library ─────────────────────────────────────────────────────────

async function loadLibrary() {
    const list1 = document.getElementById('library-list');
    const list2 = document.getElementById('library-list-2');
    if (!list1) return;

    list1.innerHTML = '<p style="padding:8px;">Loading library…</p>';

    const res = await apiFetch('/api/library');
    if (!res || !res.ok) {
        list1.innerHTML = '<p style="padding:8px;">Could not load library.</p>';
        return;
    }

    const { books } = await res.json();
    list1.innerHTML = '';
    if (list2) list2.innerHTML = '';

    if (!books || books.length === 0) {
        list1.innerHTML = '<p style="padding:8px;">Your library is empty. Add a book below!</p>';
        return;
    }

    // Split across page 2 (first 7) and page 3 (overflow)
    books.forEach((book, i) => {
        const card = buildBookCard(book.title, null, null);
        if (i < 7) {
            list1.appendChild(card);
        } else if (list2) {
            list2.appendChild(card);
        }
    });
}

// ── Add book form ─────────────────────────────────────────────────────────────

const addBookForm = document.getElementById('add-book-form');
if (addBookForm) {
    addBookForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const title = document.getElementById('add-book-title').value.trim();
        const body  = document.getElementById('add-book-body').value.trim();
        const msgEl = document.getElementById('add-book-msg');

        msgEl.style.display = 'none';

        const res = await apiFetch('/api/library', {
            method: 'POST',
            body: JSON.stringify({ title, body }),
        });
        if (!res) return;

        const data = await res.json();
        if (!res.ok) {
            msgEl.style.color = '#ff6b6b';
            msgEl.textContent = data.error || 'Failed to add book.';
        } else {
            msgEl.style.color = '#4caf50';
            msgEl.textContent = 'Book added to your library!';
            addBookForm.reset();
            loadLibrary();  // refresh library pages
        }
        msgEl.style.display = 'block';
    });
}

// ── Book search form ──────────────────────────────────────────────────────────

const searchForm = document.getElementById('search-form');
if (searchForm) {
    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const q = document.getElementById('search-query').value.trim();
        const resultsEl = document.getElementById('search-results');
        resultsEl.innerHTML = '<p style="padding:8px;">Searching…</p>';

        const res = await apiFetch(`/api/books/search?q=${encodeURIComponent(q)}`);
        if (!res || !res.ok) {
            resultsEl.innerHTML = '<p style="padding:8px;">Search failed.</p>';
            return;
        }

        const { results } = await res.json();
        resultsEl.innerHTML = '';

        if (!results || results.length === 0) {
            resultsEl.innerHTML = '<p style="padding:8px;">No books found.</p>';
            return;
        }

        results.forEach(book => {
            const card = buildBookCard(
                book.title,
                null,
                book.author ? `by ${book.author}` : (book.source === 'user' ? 'User added' : null),
            );
            resultsEl.appendChild(card);
        });
    });
}

// ── Story generation form ─────────────────────────────────────────────────────

const storyForm      = document.getElementById('story-form');
const storyContainer = document.getElementById('story-container');
const storyTitleEl   = document.getElementById('story-title');
const storyTextEl    = document.getElementById('story-text');
const saveStoryBtn   = document.getElementById('save-story-btn');
const storySaveMsg   = document.getElementById('story-save-msg');

if (storyForm) {
    storyForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const topic = document.getElementById('story-topic').value.trim();

        if (storyContainer) storyContainer.style.display = 'none';
        if (saveStoryBtn)   saveStoryBtn.style.display   = 'none';
        if (storySaveMsg)   storySaveMsg.style.display   = 'none';

        // Show a loading message
        if (storyTitleEl) storyTitleEl.textContent  = 'Generating…';
        if (storyTextEl)  storyTextEl.textContent   = '';
        if (storyContainer) storyContainer.style.display = 'block';

        const res = await apiFetch('/api/story/generate', {
            method: 'POST',
            body: JSON.stringify({ topic }),
        });
        if (!res || !res.ok) {
            if (storyTitleEl) storyTitleEl.textContent = 'Error';
            if (storyTextEl)  storyTextEl.textContent  = 'Could not generate story. Please try again.';
            return;
        }

        const data = await res.json();
        if (storyTitleEl) storyTitleEl.textContent = data.title || 'Story';
        if (storyTextEl)  storyTextEl.textContent  = data.story || '';
        if (saveStoryBtn) saveStoryBtn.style.display = 'inline-block';
    });
}

if (saveStoryBtn) {
    saveStoryBtn.addEventListener('click', async () => {
        saveStoryBtn.disabled = true;
        if (storySaveMsg) storySaveMsg.style.display = 'none';

        const res = await apiFetch('/api/story/save', { method: 'POST' });
        if (!res) return;

        const data = await res.json();
        if (!res.ok) {
            if (storySaveMsg) {
                storySaveMsg.style.color   = '#ff6b6b';
                storySaveMsg.textContent   = data.error || 'Could not save story.';
                storySaveMsg.style.display = 'block';
            }
            saveStoryBtn.disabled = false;
        } else {
            if (storySaveMsg) {
                storySaveMsg.style.color   = '#4caf50';
                storySaveMsg.textContent   = 'Story saved!';
                storySaveMsg.style.display = 'block';
            }
            saveStoryBtn.style.display = 'none';
        }
    });
}

// ── Bootstrap: load data when the page opens ──────────────────────────────────

loadProfile();
loadRecommendations();
loadLibrary();
